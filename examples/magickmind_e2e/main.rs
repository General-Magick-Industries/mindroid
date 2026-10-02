//! End-to-end check of a robot tool loop on a live MagickMind platform.
//!
//! Three end users share one fresh magickspace: a person who asks for an
//! action, a device that advertises and executes tools, and a real mindroid
//! agent backed by a scripted in-process LLM, so every run is deterministic.
//! Each hop is asserted on the wire, and a failure names the service that
//! broke it. No hardware and no real model are involved.
//!
//! Run:
//!   MM_EMAIL=you@example.com MM_PASSWORD=... \
//!     cargo run -p mindroid-example-magickmind-e2e --bin magickmind_e2e
//!
//! Env: MM_BASE_URL (default https://dev-bifrost.magickmind.ai),
//! MM_CENTRIFUGO_URL (default wss://dev-centrifugo.magickmind.ai/connection/websocket),
//! MM_E2E_PREFIX (fixture external-id prefix, default `mindroid-e2e`),
//! MM_E2E_PROJECT_ID, MM_E2E_WAIT_SECS (per-hop wait, default 30),
//! MM_E2E_KEEP_SPACE=1 to keep the magickspace, MM_E2E_REPORT=<path> for a JSON report.
//! Exits 0 when every hop passes, 1 when any fails, 2 when setup fails.

mod agent;
mod platform;
mod stub_llm;
mod wire;

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use serde::Serialize;
use serde_json::{Value, json};

use platform::{Bifrost, Reply};
use stub_llm::{StubLlm, last_user_text, message_text};
use wire::WireObserver;

const DEFAULT_BASE_URL: &str = "https://dev-bifrost.magickmind.ai";
const DEFAULT_WS_URL: &str = "wss://dev-centrifugo.magickmind.ai/connection/websocket";
const REMOTE_CALL_TIMEOUT_SECS: u64 = 15;

struct Settings {
    base_url: String,
    ws_url: String,
    email: String,
    password: String,
    prefix: String,
    project_id: String,
    wait: Duration,
    keep_space: bool,
    report_path: Option<String>,
}

impl Settings {
    fn from_env() -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let prefix = var("MM_E2E_PREFIX").unwrap_or_else(|| "mindroid-e2e".into());
        Ok(Self {
            base_url: var("MM_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            ws_url: var("MM_CENTRIFUGO_URL").unwrap_or_else(|| DEFAULT_WS_URL.into()),
            email: var("MM_EMAIL").context("MM_EMAIL is required")?,
            password: var("MM_PASSWORD").context("MM_PASSWORD is required")?,
            project_id: var("MM_E2E_PROJECT_ID").unwrap_or_else(|| prefix.clone()),
            prefix,
            wait: Duration::from_secs(
                var("MM_E2E_WAIT_SECS")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
            ),
            keep_space: var("MM_E2E_KEEP_SPACE").is_some_and(|v| v != "0"),
            report_path: var("MM_E2E_REPORT"),
        })
    }
}

#[derive(Serialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Status {
    Pass,
    Fail,
    Skip,
}

#[derive(Serialize)]
struct Hop {
    id: &'static str,
    name: &'static str,
    owner: String,
    status: Status,
    detail: String,
    traces: Vec<String>,
}

#[derive(Serialize, Default)]
struct Report {
    run: String,
    magickspace_id: String,
    hops: Vec<Hop>,
}

impl Report {
    fn record(
        &mut self,
        id: &'static str,
        name: &'static str,
        owner: impl Into<String>,
        outcome: std::result::Result<String, String>,
        traces: &[&str],
    ) -> bool {
        let (status, detail) = match outcome {
            Ok(detail) => (Status::Pass, detail),
            Err(detail) => (Status::Fail, detail),
        };
        self.hops.push(Hop {
            id,
            name,
            owner: owner.into(),
            status,
            detail,
            traces: traces.iter().map(|t| t.to_string()).collect(),
        });
        status == Status::Pass
    }

    fn skip(&mut self, id: &'static str, name: &'static str, why: &str) {
        self.hops.push(Hop {
            id,
            name,
            owner: String::new(),
            status: Status::Skip,
            detail: why.into(),
            traces: Vec::new(),
        });
    }

    fn passed(&self) -> bool {
        self.hops.iter().all(|h| h.status == Status::Pass)
    }

    fn print(&self) {
        println!("\nrun {}  magickspace {}\n", self.run, self.magickspace_id);
        for h in &self.hops {
            let mark = match h.status {
                Status::Pass => "PASS",
                Status::Fail => "FAIL",
                Status::Skip => "SKIP",
            };
            println!("[{mark}] {:<3} {}", h.id, h.name);
            if h.status != Status::Pass {
                if !h.owner.is_empty() {
                    println!("         owner: {}", h.owner);
                }
                println!("         {}", h.detail);
                if !h.traces.is_empty() {
                    println!("         trace: {}", h.traces.join(", "));
                }
            }
        }
        let count = |s| self.hops.iter().filter(|h| h.status == s).count();
        println!(
            "\n{} passed, {} failed, {} skipped",
            count(Status::Pass),
            count(Status::Fail),
            count(Status::Skip)
        );
    }
}

struct Actor {
    id: String,
    token: String,
}

struct Space {
    id: String,
    person: Actor,
    device: Actor,
    agent: Actor,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "warn,magickmind_e2e=info".into()),
        )
        .init();

    let settings = match Settings::from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("setup failed: {e:#}");
            return ExitCode::from(2);
        }
    };
    let run = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    let bifrost = match Bifrost::new(&settings.base_url) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("setup failed: {e:#}");
            return ExitCode::from(2);
        }
    };
    let jwt = match bifrost.login(&settings.email, &settings.password).await {
        Ok(jwt) => jwt,
        Err(e) => {
            eprintln!("setup failed: {e:#}");
            return ExitCode::from(2);
        }
    };
    let space = match provision(&bifrost, &jwt, &settings, &run).await {
        Ok(space) => space,
        Err(e) => {
            eprintln!("setup failed: {e:#}");
            return ExitCode::from(2);
        }
    };

    let outcome = exercise(&bifrost, &settings, &space, &run).await;

    if settings.keep_space {
        println!("keeping magickspace {}", space.id);
    } else if let Err(e) = bifrost.delete_space(&jwt, &space.id).await {
        eprintln!("cleanup: {e:#}");
    }

    let report = match outcome {
        Ok(report) => report,
        Err(e) => {
            eprintln!("setup failed: {e:#}");
            return ExitCode::from(2);
        }
    };
    report.print();
    if let Some(path) = &settings.report_path {
        match serde_json::to_vec_pretty(&report) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    eprintln!("writing {path}: {e}");
                }
            }
            Err(e) => eprintln!("serializing the report: {e}"),
        }
    }
    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

async fn provision(bifrost: &Bifrost, jwt: &str, s: &Settings, run: &str) -> Result<Space> {
    let fixture = |role: &str| format!("{}-{role}", s.prefix);
    let person = bifrost
        .ensure_end_user(jwt, &fixture("person"), "E2E person", "HUMAN")
        .await?;
    let device = bifrost
        .ensure_end_user(jwt, &fixture("device"), "E2E device", "HUMAN")
        .await?;
    let agent = bifrost
        .ensure_end_user(jwt, &fixture("agent"), "E2E agent", "AGENT")
        .await?;
    let id = bifrost
        .create_space(
            jwt,
            &format!("{} {run}", s.prefix),
            &s.project_id,
            &[&person, &device, &agent],
        )
        .await?;
    let actor = |id: String| async {
        let token = bifrost.mint(jwt, &id).await?;
        anyhow::Ok(Actor { id, token })
    };
    Ok(Space {
        id,
        person: actor(person).await?,
        device: actor(device).await?,
        agent: actor(agent).await?,
    })
}

fn field<'a>(m: &'a Value, key: &str) -> &'a str {
    m[key].as_str().unwrap_or("")
}

fn tool_call_of(m: &Value) -> Option<Value> {
    let envelope: Value = serde_json::from_str(field(m, "content")).ok()?;
    (envelope["type"] == "tool_call" && envelope["payload"]["name"] == "drive")
        .then(|| envelope["payload"].clone())
}

fn sent(reply: &Reply, what: &str) -> std::result::Result<String, String> {
    if reply.status.is_success() {
        Ok(field(&reply.body, "id").to_string())
    } else {
        Err(format!(
            "{what}: Bifrost answered {} {}",
            reply.status, reply.body
        ))
    }
}

fn expect_type(m: &Value, want: &str) -> std::result::Result<String, String> {
    match field(m, "message_type") {
        t if t == want => Ok(format!("message_type {want}")),
        "" => Err(format!("arrived with no message_type (want {want})")),
        t => Err(format!("arrived as {t} (want {want})")),
    }
}

async fn wait_for_request(
    stub: &StubLlm,
    within: Duration,
    matches: impl Fn(&Value) -> bool,
) -> Option<Value> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        if let Some(hit) = stub.requests().into_iter().find(|r| matches(r)) {
            return Some(hit);
        }
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn exercise(bifrost: &Bifrost, s: &Settings, space: &Space, run: &str) -> Result<Report> {
    let mut report = Report {
        run: run.to_string(),
        magickspace_id: space.id.clone(),
        ..Report::default()
    };
    let req_marker = format!("REQ-{run}");
    let res_marker = format!("RES-{run}");
    let toolset_marker = format!("TOOLSET-{run}");

    let script: stub_llm::Script = {
        let (req_marker, res_marker, run) =
            (req_marker.clone(), res_marker.clone(), run.to_string());
        Arc::new(move |request| {
            let last = last_user_text(request);
            if last.contains("<tool_result") {
                if last.contains(&res_marker) {
                    format!("Done, I drove forward. FINAL-OK-{run}")
                } else {
                    format!("The drive did not complete. FINAL-ERROR-{run}")
                }
            } else if last.contains(&req_marker) {
                "On it.\n<tool_call>{\"name\":\"drive\",\"args\":{\"direction\":\"forward\",\"seconds\":1,\"note\":\"R&D bench\"}}</tool_call>".into()
            } else {
                format!("UNEXPECTED-TURN-{run}")
            }
        })
    };
    let stub = StubLlm::start(script).await?;

    let mut person_wire = WireObserver::connect(&s.ws_url, &space.person.token, "person").await?;
    let mut device_wire = WireObserver::connect(&s.ws_url, &space.device.token, "device").await?;
    let mut agent_wire = WireObserver::connect(&s.ws_url, &space.agent.token, "agent").await?;

    let agent = agent::start(agent::AgentSetup {
        base_url: s.base_url.clone(),
        ws_url: s.ws_url.clone(),
        agent_id: space.agent.id.clone(),
        token: space.agent.token.clone(),
        device_id: space.device.id.clone(),
        llm_base_url: stub.base_url.clone(),
    })
    .await?;

    let outcome = run_hops(
        &mut report,
        bifrost,
        s,
        space,
        &stub,
        (&mut person_wire, &mut device_wire, &mut agent_wire),
        (&req_marker, &res_marker, &toolset_marker),
        run,
    )
    .await;
    agent.stop().await;
    outcome?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
async fn run_hops(
    report: &mut Report,
    bifrost: &Bifrost,
    s: &Settings,
    space: &Space,
    stub: &StubLlm,
    (person_wire, device_wire, agent_wire): (
        &mut WireObserver,
        &mut WireObserver,
        &mut WireObserver,
    ),
    (req_marker, res_marker, toolset_marker): (&str, &str, &str),
    run: &str,
) -> Result<()> {
    let wait = s.wait;
    let tools = json!([{
        "name": "drive",
        "description": format!("Drive the robot base. {toolset_marker}"),
        "schema": {
            "type": "object",
            "properties": {
                "direction": { "type": "string", "enum": ["forward", "backward"] },
                "seconds": { "type": "number" },
                "note": { "type": "string" },
            },
            "required": ["direction"],
        },
        "timeout_secs": REMOTE_CALL_TIMEOUT_SECS,
    }]);

    let bare = bifrost
        .send(
            &space.device.token,
            &space.id,
            &json!({ "message_type": "TOOL_MANIFEST", "content": "", "tools": tools }),
        )
        .await?;
    let bare_trace = bare.trace_id.clone();
    let mut manifest_traces = vec![bare_trace.clone()];
    if !report.record(
        "2",
        "device's TOOL_MANIFEST with no content is accepted",
        "bifrost (send validation)",
        sent(&bare, "content-less manifest").map(|_| "accepted".into()),
        &[&bare_trace],
    ) {
        let padded = bifrost
            .send(
                &space.device.token,
                &space.id,
                &json!({ "message_type": "TOOL_MANIFEST", "content": "tool manifest", "tools": tools }),
            )
            .await?;
        manifest_traces.push(padded.trace_id.clone());
        if let Err(e) = sent(&padded, "manifest with content") {
            report.record(
                "2a",
                "TOOL_MANIFEST reaches the agent typed",
                "bifrost",
                Err(e),
                &[&padded.trace_id],
            );
            for (id, name) in [
                ("1", "person's TEXT reaches the agent"),
                ("3", "agent's TOOL_CALL reaches the device typed"),
            ] {
                report.skip(id, name, "no manifest was accepted");
            }
            return Ok(());
        }
    }

    let device_id = space.device.id.clone();
    let manifest_seen = agent_wire
        .expect(wait, |m| {
            field(m, "sent_by_user_id") == device_id && m["tools"][0]["name"] == "drive"
        })
        .await;
    let traces: Vec<&str> = manifest_traces.iter().map(String::as_str).collect();
    report.record(
        "2a",
        "TOOL_MANIFEST reaches the agent typed, with its tools",
        "bifrost (fan-out)",
        match &manifest_seen {
            Some(m) => expect_type(m, "TOOL_MANIFEST"),
            None => Err(format!(
                "nothing carrying the tools reached the agent within {wait:?}"
            )),
        },
        &traces,
    );
    tokio::time::sleep(Duration::from_secs(2)).await;

    let ask = format!("Please drive forward. {req_marker}");
    let asked = bifrost
        .send(
            &space.person.token,
            &space.id,
            &json!({ "message_type": "TEXT", "content": ask }),
        )
        .await?;
    let ask_id = match sent(&asked, "person's TEXT") {
        Ok(id) => id,
        Err(e) => {
            report.record(
                "1",
                "person's TEXT reaches the agent",
                "bifrost (send)",
                Err(e),
                &[&asked.trace_id],
            );
            return Ok(());
        }
    };
    let person_id = space.person.id.clone();
    let delivered = agent_wire
        .expect(wait, |m| {
            field(m, "id") == ask_id || field(m, "content") == ask
        })
        .await;
    report.record(
        "1",
        "person's TEXT reaches the agent with sender and space intact",
        "bifrost (fan-out)",
        match &delivered {
            None => Err(format!(
                "nothing reached the agent's channel within {wait:?}"
            )),
            Some(m) => {
                let mut wrong = Vec::new();
                if field(m, "sent_by_user_id") != person_id {
                    wrong.push(format!("sent_by_user_id {:?}", field(m, "sent_by_user_id")));
                }
                if field(m, "magickspace_id") != space.id {
                    wrong.push(format!("magickspace_id {:?}", field(m, "magickspace_id")));
                }
                if field(m, "content") != ask {
                    wrong.push("content changed".into());
                }
                if let Err(e) = expect_type(m, "TEXT") {
                    wrong.push(e);
                }
                if wrong.is_empty() {
                    Ok("intact".into())
                } else {
                    Err(wrong.join("; "))
                }
            }
        },
        &[&asked.trace_id],
    );

    let first_turn = wait_for_request(stub, wait, |r| last_user_text(r).contains(req_marker)).await;
    report.record(
        "1b",
        "agent runs a turn on the person's message",
        "mindroid (transport / runtime)",
        first_turn
            .as_ref()
            .map(|_| "the LLM was called with the request".to_string())
            .ok_or_else(|| format!("the agent never called its LLM within {wait:?}")),
        &[],
    );
    report.record(
        "2b",
        "manifest tools reach the agent's registry",
        "mindroid (ManifestStage)",
        match &first_turn {
            None => Err("no agent turn to inspect".into()),
            Some(r) => {
                let prompt: String = r["messages"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|m| m["role"] == "system")
                    .map(message_text)
                    .collect();
                if prompt.contains(toolset_marker) {
                    Ok("the drive tool is in the system prompt".into())
                } else {
                    Err("the system prompt does not offer the manifest's drive tool".into())
                }
            }
        },
        &[],
    );

    let agent_id = space.agent.id.clone();
    let call_msg = device_wire
        .expect(wait, |m| {
            field(m, "sent_by_user_id") == agent_id && tool_call_of(m).is_some()
        })
        .await;
    let Some(call_msg) = call_msg else {
        report.record(
            "3",
            "agent's TOOL_CALL reaches the device typed",
            "mindroid (send) / bifrost (fan-out)",
            Err(format!(
                "no drive tool call from the agent reached the device within {wait:?}"
            )),
            &[],
        );
        for (id, name) in [
            ("4", "device's TOOL_RESULT reaches the agent typed"),
            ("4b", "agent correlates the result without timing out"),
            ("5", "reloaded history keeps message_type"),
            (
                "5b",
                "agent replays its own call unescaped, and the result once",
            ),
        ] {
            report.skip(id, name, "no tool call to answer");
        }
        return Ok(());
    };
    let call_typed = expect_type(&call_msg, "TOOL_CALL");
    let call_id = tool_call_of(&call_msg)
        .and_then(|p| p["tool_call_id"].as_str().map(str::to_string))
        .unwrap_or_default();
    let call_msg_id = field(&call_msg, "id").to_string();

    let result_body = json!({
        "name": "drive",
        "content": format!("drove forward {res_marker}"),
        "tool_call_id": call_id,
    })
    .to_string();
    let answered = bifrost
        .send(
            &space.device.token,
            &space.id,
            &json!({
                "message_type": "TOOL_RESULT",
                "content": result_body,
                "reply_to_message_id": call_msg_id,
            }),
        )
        .await?;
    let result_typed = match sent(&answered, "device's TOOL_RESULT") {
        Err(e) => Err(e),
        Ok(result_id) => {
            let echoed = agent_wire
                .expect(wait, |m| field(m, "id") == result_id)
                .await;
            match echoed {
                None => Err(format!(
                    "the result never reached the agent's channel within {wait:?}"
                )),
                Some(m) => expect_type(&m, "TOOL_RESULT"),
            }
        }
    };
    let result_id = field(&answered.body, "id").to_string();

    let call_owner = if result_typed.is_ok() {
        "mindroid (the agent sends no message_type)"
    } else {
        "bifrost (fan-out drops types), and possibly mindroid"
    };
    report.record(
        "3",
        "agent's TOOL_CALL reaches the device typed",
        call_owner,
        call_typed,
        &[],
    );
    report.record(
        "4",
        "device's TOOL_RESULT reaches the agent typed",
        "bifrost (fan-out) / chathistory (stored type)",
        result_typed,
        &[&answered.trace_id],
    );

    let final_wait = wait + Duration::from_secs(REMOTE_CALL_TIMEOUT_SECS);
    let final_reply = person_wire
        .expect(final_wait, |m| {
            field(m, "sent_by_user_id") == agent_id
                && ["FINAL-OK-", "FINAL-ERROR-", "UNEXPECTED-TURN-"]
                    .iter()
                    .any(|tag| field(m, "content").contains(&format!("{tag}{run}")))
        })
        .await;
    report.record(
        "4b",
        "agent correlates the result without timing out",
        "mindroid (result gate) — fails downstream of hop 4",
        match &final_reply {
            None => Err(format!("the agent said nothing within {final_wait:?}")),
            Some(m) => {
                let content = field(m, "content");
                if content.contains(&format!("FINAL-OK-{run}")) {
                    Ok("the agent finished the turn with the device's result".into())
                } else if content.contains(&format!("FINAL-ERROR-{run}")) {
                    Err(
                        "the call timed out: the agent reported a failure for a completed action"
                            .into(),
                    )
                } else {
                    Err("the result reached the model as plain chat, uncorrelated".into())
                }
            }
        },
        &[],
    );

    let want = [
        (ask_id.as_str(), "TEXT", "person's request"),
        (call_msg_id.as_str(), "TOOL_CALL", "agent's call"),
        (result_id.as_str(), "TOOL_RESULT", "device's result"),
    ];
    let check_history = |items: &[Value]| -> std::result::Result<String, String> {
        let wrong: Vec<String> = want
            .iter()
            .filter(|(id, _, _)| !id.is_empty())
            .filter_map(
                |(id, ty, what)| match items.iter().find(|m| field(m, "id") == *id) {
                    None => Some(format!("{what} is missing")),
                    Some(m) => expect_type(m, ty).err().map(|e| format!("{what} {e}")),
                },
            )
            .collect();
        if wrong.is_empty() {
            Ok("every message kept its type".into())
        } else {
            Err(wrong.join("; "))
        }
    };

    let listed = bifrost
        .list_messages(&space.person.token, &space.id)
        .await?;
    report.record(
        "5",
        "reloaded history keeps message_type",
        "chathistory (stored type)",
        if listed.status.is_success() {
            check_history(
                listed.body["data"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            )
        } else {
            Err(format!(
                "listing messages: {} {}",
                listed.status, listed.body
            ))
        },
        &[&listed.trace_id],
    );
    let prepared = bifrost
        .prepare_context(&space.agent.token, &space.id)
        .await?;
    report.record(
        "5a",
        "context prepare keeps message_type",
        "chathistory (stored type) / bifrost (context prepare)",
        if prepared.status.is_success() {
            check_history(
                prepared.body["chat_history"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            )
        } else {
            Err(format!(
                "context prepare: {} {}",
                prepared.status, prepared.body
            ))
        },
        &[&prepared.trace_id],
    );

    let result_turn = stub.requests().into_iter().find(|r| {
        last_user_text(r).contains("<tool_result") && last_user_text(r).contains(res_marker)
    });
    match result_turn {
        None => report.skip(
            "5b",
            "agent replays its own call unescaped, and the result once",
            "the agent never ran a correlated result turn (see 4b)",
        ),
        Some(r) => {
            let messages = r["messages"].as_array().cloned().unwrap_or_default();
            let own_call = messages
                .iter()
                .filter(|m| m["role"] == "assistant")
                .map(message_text)
                .find(|t| t.contains("drive"));
            let result_copies = messages
                .iter()
                .map(message_text)
                .filter(|t| t.contains(res_marker))
                .count();
            let mut wrong = Vec::new();
            match &own_call {
                None => wrong.push("the agent's own call is missing from its history".to_string()),
                Some(t)
                    if ["&amp;", "&lt;", "&gt;", "&quot;"]
                        .iter()
                        .any(|entity| t.contains(entity)) =>
                {
                    wrong.push("the agent's own call replays HTML-escaped".into())
                }
                Some(_) => {}
            }
            if result_copies != 1 {
                wrong.push(format!(
                    "the device's result appears {result_copies} times in the prompt"
                ));
            }
            report.record(
                "5b",
                "agent replays its own call unescaped, and the result once",
                "mindroid (history replay)",
                if wrong.is_empty() {
                    Ok("replayed faithfully".into())
                } else {
                    Err(wrong.join("; "))
                },
                &[],
            );
        }
    }

    Ok(())
}
