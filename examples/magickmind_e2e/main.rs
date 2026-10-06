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
//! MM_E2E_KEEP_SPACE=1 to keep the magickspace, MM_E2E_REPORT=<path> for a JSON report,
//! MM_E2E_ALLOW_INSECURE=1 to allow http/ws URLs for a local stack.
//!
//! The fixtures are shared end users, so runs that overlap in one tenant need
//! distinct MM_E2E_PREFIX values; otherwise each run's agent answers the other's space.
//!
//! Exits 0 when every hop passes, 1 when any fails (including Bifrost going
//! unreachable mid-run), 2 when setup fails before the first hop.

mod agent;
mod platform;
mod stub_llm;
mod wire;

use std::{process::ExitCode, sync::Arc, time::Duration};

use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use agent::AgentHandle;
use platform::{Bifrost, Reply};
use stub_llm::{StubLlm, last_user_text, message_text};
use wire::WireObserver;

const DEFAULT_BASE_URL: &str = "https://dev-bifrost.magickmind.ai";
const DEFAULT_WS_URL: &str = "wss://dev-centrifugo.magickmind.ai/connection/websocket";
const REMOTE_CALL_TIMEOUT_SECS: u64 = 15;

type Outcome = std::result::Result<String, String>;

#[derive(Debug)]
struct HopSpec {
    id: &'static str,
    name: &'static str,
    owner: &'static str,
}

const HOPS: &[HopSpec] = &[
    HopSpec {
        id: "2",
        name: "device's TOOL_MANIFEST with no content is accepted",
        owner: "bifrost (send validation)",
    },
    HopSpec {
        id: "2a",
        name: "TOOL_MANIFEST reaches the agent typed, with its tools",
        owner: "bifrost (fan-out)",
    },
    HopSpec {
        id: "1",
        name: "person's TEXT reaches the agent with sender and space intact",
        owner: "bifrost (fan-out)",
    },
    HopSpec {
        id: "1b",
        name: "agent runs a turn on the person's message",
        owner: "mindroid (transport / runtime)",
    },
    HopSpec {
        id: "2b",
        name: "manifest tools reach the agent's registry",
        owner: "mindroid (ManifestStage)",
    },
    HopSpec {
        id: "3",
        name: "agent's TOOL_CALL reaches the device typed",
        owner: "mindroid (send) / bifrost (fan-out)",
    },
    HopSpec {
        id: "4",
        name: "device's TOOL_RESULT reaches the agent typed",
        owner: "bifrost (fan-out) / chathistory (stored type)",
    },
    HopSpec {
        id: "4b",
        name: "agent correlates the result without timing out",
        owner: "mindroid (result gate)",
    },
    HopSpec {
        id: "5",
        name: "reloaded history keeps message_type",
        owner: "chathistory (stored type)",
    },
    HopSpec {
        id: "5a",
        name: "context prepare keeps message_type",
        owner: "chathistory (stored type) / bifrost (context prepare)",
    },
    HopSpec {
        id: "5b",
        name: "agent replays its own call unescaped, and the result once",
        owner: "mindroid (history replay)",
    },
];

fn spec(id: &str) -> &'static HopSpec {
    HOPS.iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("hop {id} is not in HOPS"))
}

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
    allow_insecure: bool,
}

impl Settings {
    fn from_env() -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let prefix = var("MM_E2E_PREFIX").unwrap_or_else(|| "mindroid-e2e".into());
        ensure!(
            prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
            "MM_E2E_PREFIX may only hold letters, digits, '-' and '_'"
        );
        let base_url = var("MM_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into());
        let ws_url = var("MM_CENTRIFUGO_URL").unwrap_or_else(|| DEFAULT_WS_URL.into());
        let allow_insecure = var("MM_E2E_ALLOW_INSECURE").is_some_and(|v| v != "0");
        if !allow_insecure {
            require_scheme("MM_BASE_URL", &base_url, "https")?;
            require_scheme("MM_CENTRIFUGO_URL", &ws_url, "wss")?;
        }
        let wait_secs = var("MM_E2E_WAIT_SECS")
            .map(|v| {
                v.parse::<u64>()
                    .with_context(|| format!("MM_E2E_WAIT_SECS {v:?} is not a whole number"))
            })
            .transpose()?
            .unwrap_or(30);
        Ok(Self {
            base_url,
            ws_url,
            email: var("MM_EMAIL").context("MM_EMAIL is required")?,
            password: var("MM_PASSWORD").context("MM_PASSWORD is required")?,
            project_id: var("MM_E2E_PROJECT_ID").unwrap_or_else(|| prefix.clone()),
            prefix,
            wait: Duration::from_secs(wait_secs),
            keep_space: var("MM_E2E_KEEP_SPACE").is_some_and(|v| v != "0"),
            report_path: var("MM_E2E_REPORT"),
            allow_insecure,
        })
    }
}

fn require_scheme(var: &str, url: &str, scheme: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("{var} {url:?} is not a URL"))?;
    ensure!(
        parsed.scheme() == scheme,
        "{var} {url} is not {scheme}://; set MM_E2E_ALLOW_INSECURE=1 to run against a local stack"
    );
    Ok(())
}

#[derive(Serialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum Status {
    Pass,
    Fail,
    Skip,
}

#[derive(Serialize, Debug)]
struct Hop {
    id: &'static str,
    name: &'static str,
    owner: String,
    status: Status,
    detail: String,
    traces: Vec<String>,
    messages: Vec<String>,
}

impl Hop {
    fn trace(&mut self, id: &str) -> &mut Self {
        if !id.is_empty() {
            self.traces.push(id.to_string());
        }
        self
    }

    fn message(&mut self, id: &str) -> &mut Self {
        if !id.is_empty() {
            self.messages.push(id.to_string());
        }
        self
    }

    fn passed(&self) -> bool {
        self.status == Status::Pass
    }
}

#[derive(Serialize, Default, Debug)]
struct Report {
    run: String,
    magickspace_id: String,
    hops: Vec<Hop>,
}

impl Report {
    fn push(&mut self, id: &str, status: Status, owner: &str, detail: String) -> &mut Hop {
        let spec = spec(id);
        self.hops.push(Hop {
            id: spec.id,
            name: spec.name,
            owner: owner.to_string(),
            status,
            detail,
            traces: Vec::new(),
            messages: Vec::new(),
        });
        let last = self.hops.len() - 1;
        &mut self.hops[last]
    }

    fn record(&mut self, id: &str, outcome: Outcome) -> &mut Hop {
        let owner = spec(id).owner;
        match outcome {
            Ok(detail) => self.push(id, Status::Pass, owner, detail),
            Err(detail) => self.push(id, Status::Fail, owner, detail),
        }
    }

    fn skip(&mut self, id: &str, why: &str) {
        self.push(id, Status::Skip, "", why.to_string());
    }

    fn hop(&mut self, id: &str) -> Option<&mut Hop> {
        self.hops.iter_mut().find(|h| h.id == id)
    }

    fn finish(&mut self, why: &str) {
        let missing: Vec<&str> = HOPS
            .iter()
            .map(|s| s.id)
            .filter(|id| !self.hops.iter().any(|h| h.id == *id))
            .collect();
        for id in missing {
            self.skip(id, why);
        }
        self.hops
            .sort_by_key(|h| HOPS.iter().position(|s| s.id == h.id));
    }

    fn passed(&self) -> bool {
        self.hops.iter().all(Hop::passed)
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
            if h.passed() {
                continue;
            }
            if !h.owner.is_empty() {
                println!("         owner: {}", h.owner);
            }
            println!("         {}", h.detail);
            if !h.traces.is_empty() {
                println!("         trace: {}", h.traces.join(", "));
            }
            if !h.messages.is_empty() {
                println!("         messages: {}", h.messages.join(", "));
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

    fn write(&self, path: &str) {
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    eprintln!("writing {path}: {e}");
                }
            }
            Err(e) => eprintln!("serializing the report: {e}"),
        }
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

struct Session {
    settings: Settings,
    bifrost: Bifrost,
    jwt: String,
    space: Space,
    run: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "warn,magickmind_e2e=info".into()),
        )
        .init();

    let Session {
        settings,
        bifrost,
        jwt,
        space,
        run,
    } = match setup().await {
        Ok(session) => session,
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
        report.write(path);
    }
    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

async fn setup() -> Result<Session> {
    let settings = Settings::from_env()?;
    let run = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    let bifrost = Bifrost::new(&settings.base_url)?;
    let jwt = bifrost.login(&settings.email, &settings.password).await?;
    let space = provision(&bifrost, &jwt, &settings, &run).await?;
    Ok(Session {
        settings,
        bifrost,
        jwt,
        space,
        run,
    })
}

async fn provision(bifrost: &Bifrost, jwt: &str, s: &Settings, run: &str) -> Result<Space> {
    let actor = |role: &'static str, name: &'static str, kind: &'static str| async move {
        let id = bifrost
            .ensure_end_user(jwt, &format!("{}-{role}", s.prefix), name, kind)
            .await?;
        let token = bifrost.mint(jwt, &id).await?;
        anyhow::Ok(Actor { id, token })
    };
    let person = actor("person", "E2E person", "HUMAN").await?;
    let device = actor("device", "E2E device", "HUMAN").await?;
    let agent = actor("agent", "E2E agent", "AGENT").await?;
    let id = bifrost
        .create_space(
            jwt,
            &format!("{} {run}", s.prefix),
            &s.project_id,
            &[&person.id, &device.id, &agent.id],
        )
        .await?;
    Ok(Space {
        id,
        person,
        device,
        agent,
    })
}

struct Markers {
    req: String,
    res: String,
    toolset: String,
}

impl Markers {
    fn new(run: &str) -> Self {
        Self {
            req: format!("REQ-{run}"),
            res: format!("RES-{run}"),
            toolset: format!("TOOLSET-{run}"),
        }
    }
}

fn script(markers: &Markers, run: &str) -> stub_llm::Script {
    let (req, res, run) = (markers.req.clone(), markers.res.clone(), run.to_string());
    Arc::new(move |request| {
        let last = last_user_text(request);
        if last.contains("<tool_result") {
            if last.contains(&res) {
                format!("Done, I drove forward. FINAL-OK-{run}")
            } else {
                format!("The drive did not complete. FINAL-ERROR-{run}")
            }
        } else if last.contains(&req) {
            "On it.\n<tool_call>{\"name\":\"drive\",\"args\":{\"direction\":\"forward\",\"seconds\":1,\"note\":\"R&D bench\"}}</tool_call>".into()
        } else {
            format!("UNEXPECTED-TURN-{run}")
        }
    })
}

async fn exercise(bifrost: &Bifrost, s: &Settings, space: &Space, run: &str) -> Result<Report> {
    let markers = Markers::new(run);
    let stub = StubLlm::start(script(&markers, run)).await?;
    let wires = Wires {
        person: WireObserver::connect(&s.ws_url, &space.person.token, "person").await?,
        device: WireObserver::connect(&s.ws_url, &space.device.token, "device").await?,
        agent: WireObserver::connect(&s.ws_url, &space.agent.token, "agent").await?,
    };
    let agent = agent::start(agent::AgentSetup {
        base_url: s.base_url.clone(),
        ws_url: s.ws_url.clone(),
        agent_id: space.agent.id.clone(),
        token: space.agent.token.clone(),
        device_id: space.device.id.clone(),
        llm_base_url: stub.base_url.clone(),
        allow_insecure: s.allow_insecure,
    })
    .await?;

    let mut harness = Harness {
        bifrost,
        space,
        stub: &stub,
        agent: &agent,
        wires,
        wait: s.wait,
        run,
        markers,
        report: Report {
            run: run.to_string(),
            magickspace_id: space.id.clone(),
            ..Report::default()
        },
        halted: String::new(),
    };
    harness.hops().await;
    let (mut report, halted) = (harness.report, harness.halted);
    report.finish(if halted.is_empty() {
        "not reached"
    } else {
        &halted
    });
    agent.stop().await;
    Ok(report)
}

struct Wires {
    person: WireObserver,
    device: WireObserver,
    agent: WireObserver,
}

struct Call {
    msg_id: String,
    call_id: String,
    sent_as: String,
}

struct Harness<'a> {
    bifrost: &'a Bifrost,
    space: &'a Space,
    stub: &'a StubLlm,
    agent: &'a AgentHandle,
    wires: Wires,
    wait: Duration,
    run: &'a str,
    markers: Markers,
    report: Report,
    halted: String,
}

impl Harness<'_> {
    async fn hops(&mut self) -> Option<()> {
        self.manifest().await?;
        let ask_id = self.ask().await?;
        let call = self.call().await?;
        let result_id = self.answer(&call).await?;
        self.history(&ask_id, &call, result_id.as_deref()).await?;
        self.replay();
        Some(())
    }

    fn halt<T>(&mut self, why: &str) -> Option<T> {
        self.halted = why.to_string();
        None
    }

    fn reach(&mut self, hop: &str, reply: Result<Reply>) -> Option<Reply> {
        match reply {
            Ok(reply) => Some(reply),
            Err(e) => {
                self.report.record(hop, Err(format!("{e:#}"))).owner =
                    "bifrost (unreachable)".into();
                self.halt("Bifrost stopped answering")
            }
        }
    }

    async fn manifest(&mut self) -> Option<()> {
        let (bifrost, space, wait) = (self.bifrost, self.space, self.wait);
        let tools = json!([{
            "name": "drive",
            "description": format!("Drive the robot base. {}", self.markers.toolset),
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
        let manifest = |content: &str| json!({ "message_type": "TOOL_MANIFEST", "content": content, "tools": tools });

        let sent_bare = bifrost
            .send(&space.device.token, &space.id, &manifest(""))
            .await;
        let bare = self.reach("2", sent_bare)?;
        let mut traces = vec![bare.trace_id.clone()];
        let bare_ok = self
            .report
            .record("2", accepted(&bare, "content-less manifest"))
            .trace(&bare.trace_id)
            .passed();
        if !bare_ok {
            let sent_padded = bifrost
                .send(&space.device.token, &space.id, &manifest("tool manifest"))
                .await;
            let padded = self.reach("2a", sent_padded)?;
            traces.push(padded.trace_id.clone());
            if let Err(e) = accepted(&padded, "manifest with content") {
                self.report.record("2a", Err(e)).trace(&padded.trace_id);
                return self.halt("no manifest was accepted");
            }
        }

        let seen = self
            .wires
            .agent
            .wait_for(wait, |m| {
                field(m, "magickspace_id") == space.id
                    && field(m, "sent_by_user_id") == space.device.id
                    && m["tools"][0]["name"] == "drive"
            })
            .await;
        let hop = self.report.record(
            "2a",
            seen.map_err(|miss| format!("nothing carrying the tools reached the agent {miss}"))
                .and_then(|m| expect_type(&m, &["TOOL_MANIFEST"])),
        );
        for t in &traces {
            hop.trace(t);
        }
        self.agent.wait_for_tool("drive", wait).await;
        Some(())
    }

    async fn ask(&mut self) -> Option<String> {
        let (bifrost, space, stub, wait) = (self.bifrost, self.space, self.stub, self.wait);
        let ask = format!("Please drive forward. {}", self.markers.req);
        let sent_ask = bifrost
            .send(
                &space.person.token,
                &space.id,
                &json!({ "message_type": "TEXT", "content": ask }),
            )
            .await;
        let asked = self.reach("1", sent_ask)?;
        let ask_id = match sent(&asked, "person's TEXT") {
            Ok(id) => id,
            Err(e) => {
                self.report.record("1", Err(e)).trace(&asked.trace_id);
                return self.halt("the person's message was refused");
            }
        };
        let delivered = self
            .wires
            .agent
            .wait_for(wait, |m| {
                field(m, "id") == ask_id || field(m, "content") == ask
            })
            .await;
        let outcome = match &delivered {
            Err(miss) => Err(format!("nothing reached the agent's channel {miss}")),
            Ok(m) => intact(m, &space.person.id, &space.id, &ask),
        };
        self.report
            .record("1", outcome)
            .trace(&asked.trace_id)
            .message(&ask_id);

        let first_turn = wait_for_request(stub, wait, |r| {
            last_user_text(r).contains(&self.markers.req)
        })
        .await;
        let Some(turn) = first_turn else {
            if delivered.is_err() {
                self.report
                    .skip("1b", "the request never reached the agent (see 1)");
            } else {
                self.report.record(
                    "1b",
                    Err(format!("the agent never called its LLM within {wait:?}")),
                );
            }
            return self.halt("the agent never ran a turn");
        };
        self.report
            .record("1b", Ok("the LLM was called with the request".into()));

        let prompt: String = turn["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["role"] == "system")
            .map(message_text)
            .collect();
        self.report.record(
            "2b",
            if prompt.contains(&self.markers.toolset) {
                Ok("the drive tool is in the system prompt".into())
            } else {
                Err("the system prompt does not offer the manifest's drive tool".into())
            },
        );
        Some(ask_id)
    }

    async fn call(&mut self) -> Option<Call> {
        let (space, wait) = (self.space, self.wait);
        let seen = self
            .wires
            .device
            .wait_for(wait, |m| {
                field(m, "magickspace_id") == space.id
                    && field(m, "sent_by_user_id") == space.agent.id
                    && tool_call_of(m).is_some()
            })
            .await;
        let m = match seen {
            Ok(m) => m,
            Err(miss) => {
                self.report.record(
                    "3",
                    Err(format!(
                        "no drive tool call from the agent reached the device {miss}"
                    )),
                );
                return self.halt("no tool call to answer");
            }
        };
        let call = Call {
            msg_id: field(&m, "id").to_string(),
            call_id: tool_call_of(&m)
                .and_then(|p| p["tool_call_id"].as_str().map(str::to_string))
                .unwrap_or_default(),
            sent_as: field(&m, "message_type").to_string(),
        };
        self.report
            .record("3", expect_type(&m, &["TOOL_CALL"]))
            .message(&call.msg_id);
        Some(call)
    }

    async fn answer(&mut self, call: &Call) -> Option<Option<String>> {
        let (bifrost, space, wait, run) = (self.bifrost, self.space, self.wait, self.run);
        let result = json!({
            "type": "tool_result",
            "payload": {
                "tool_call_id": call.call_id,
                "name": "drive",
                "content": format!("drove forward {}", self.markers.res),
            },
        })
        .to_string();
        let sent_result = bifrost
            .send(
                &space.device.token,
                &space.id,
                &json!({
                    "message_type": "TOOL_RESULT",
                    "content": result,
                    "reply_to_message_id": call.msg_id,
                }),
            )
            .await;
        let answered = self.reach("4", sent_result)?;
        let result_id = sent(&answered, "device's TOOL_RESULT");
        let arrived = match &result_id {
            Err(e) => Err(e.clone()),
            Ok(id) => self
                .wires
                .agent
                .wait_for(wait, |m| field(m, "id") == id)
                .await
                .map_err(|miss| format!("the result never reached the agent's channel {miss}")),
        };
        let typed = arrived
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|m| expect_type(m, &["TOOL_RESULT"]));

        if let Some(hop) = self.report.hop("3").filter(|h| !h.passed()) {
            hop.owner = match (&arrived, &typed) {
                (_, Ok(_)) => {
                    "mindroid (the agent sends no message_type; fan-out kept the result's)"
                }
                (Ok(_), Err(_)) => "bifrost (fan-out drops types), and possibly mindroid",
                (Err(_), _) => "mindroid (send) or bifrost (fan-out); hop 4 could not tell which",
            }
            .into();
        }

        let result_id = result_id.ok();
        let result_ok = typed.is_ok();
        let hop = self.report.record("4", typed).trace(&answered.trace_id);
        if let Some(id) = &result_id {
            hop.message(id);
        }

        let final_wait = wait + Duration::from_secs(REMOTE_CALL_TIMEOUT_SECS);
        let tagged = |content: &str, tag: &str| content.contains(&format!("{tag}{run}"));
        let reply = self
            .wires
            .person
            .wait_for(final_wait, |m| {
                field(m, "sent_by_user_id") == space.agent.id
                    && ["FINAL-OK-", "FINAL-ERROR-", "UNEXPECTED-TURN-"]
                        .iter()
                        .any(|tag| tagged(field(m, "content"), tag))
            })
            .await;
        let outcome = match &reply {
            Err(miss) => Err(format!("the agent said nothing {miss}")),
            Ok(m) if tagged(field(m, "content"), "FINAL-OK-") => {
                Ok("the agent finished the turn with the device's result".into())
            }
            Ok(m) if tagged(field(m, "content"), "FINAL-ERROR-") => Err(
                "the call timed out: the agent reported a failure for a completed action".into(),
            ),
            Ok(_) => Err("the result reached the model as plain chat, uncorrelated".into()),
        };
        let hop = self.report.record("4b", outcome).trace(&answered.trace_id);
        if let Some(id) = &result_id {
            hop.message(id);
        }
        if let Ok(m) = &reply {
            hop.message(field(m, "id"));
        }
        if !result_ok && !hop.passed() {
            hop.owner = "downstream of hop 4".into();
        }
        Some(result_id)
    }

    async fn history(&mut self, ask_id: &str, call: &Call, result_id: Option<&str>) -> Option<()> {
        let (bifrost, space) = (self.bifrost, self.space);
        let sent_as = [call.sent_as.as_str()];
        let call_types: &[&str] = if call.sent_as.is_empty() {
            &["TOOL_CALL", "TEXT"]
        } else {
            &sent_as
        };
        let want: [(&str, &[&str], &str); 3] = [
            (ask_id, &["TEXT"], "person's request"),
            (&call.msg_id, call_types, "agent's call"),
            (result_id.unwrap_or(""), &["TOOL_RESULT"], "device's result"),
        ];

        let listed = bifrost.list_messages(&space.person.token, &space.id).await;
        let listed = self.reach("5", listed)?;
        let outcome = if listed.status.is_success() {
            check_history(items(&listed.body["data"]), &want)
        } else {
            Err(format!(
                "listing messages: {} {}",
                listed.status, listed.body
            ))
        };
        self.report.record("5", outcome).trace(&listed.trace_id);

        let prepared = bifrost.prepare_context(&space.agent.token, &space.id).await;
        let prepared = self.reach("5a", prepared)?;
        let outcome = if prepared.status.is_success() {
            check_history(items(&prepared.body["chat_history"]), &want)
        } else {
            Err(format!(
                "context prepare: {} {}",
                prepared.status, prepared.body
            ))
        };
        self.report.record("5a", outcome).trace(&prepared.trace_id);
        Some(())
    }

    fn replay(&mut self) {
        let res = self.markers.res.as_str();
        let Some(turn) = self.stub.requests().into_iter().find(|r| {
            let last = last_user_text(r);
            last.contains("<tool_result") && last.contains(res)
        }) else {
            self.report.skip(
                "5b",
                "the agent never ran a correlated result turn (see 4b)",
            );
            return;
        };
        let messages = turn["messages"].as_array().cloned().unwrap_or_default();
        let own_call = messages
            .iter()
            .filter(|m| m["role"] == "assistant")
            .map(message_text)
            .find(|t| t.contains("drive"));
        let result_copies = messages
            .iter()
            .map(message_text)
            .filter(|t| t.contains(res))
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
        self.report.record(
            "5b",
            if wrong.is_empty() {
                Ok("replayed faithfully".into())
            } else {
                Err(wrong.join("; "))
            },
        );
    }
}

fn field<'a>(m: &'a Value, key: &str) -> &'a str {
    m[key].as_str().unwrap_or("")
}

fn items(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}

fn tool_call_of(m: &Value) -> Option<Value> {
    let envelope: Value = serde_json::from_str(field(m, "content")).ok()?;
    (envelope["type"] == "tool_call" && envelope["payload"]["name"] == "drive")
        .then(|| envelope["payload"].clone())
}

fn accepted(reply: &Reply, what: &str) -> Outcome {
    if reply.status.is_success() {
        Ok("accepted".into())
    } else {
        Err(format!(
            "{what}: Bifrost answered {} {}",
            reply.status, reply.body
        ))
    }
}

fn sent(reply: &Reply, what: &str) -> Outcome {
    accepted(reply, what)?;
    match field(&reply.body, "id") {
        "" => Err(format!(
            "{what}: Bifrost answered {} with no message id",
            reply.status
        )),
        id => Ok(id.to_string()),
    }
}

fn expect_type(m: &Value, want: &[&str]) -> Outcome {
    let wanted = want.join(" or ");
    match field(m, "message_type") {
        t if want.contains(&t) => Ok(format!("message_type {t}")),
        "" => Err(format!("arrived with no message_type (want {wanted})")),
        t => Err(format!("arrived as {t} (want {wanted})")),
    }
}

fn intact(m: &Value, sender: &str, space: &str, content: &str) -> Outcome {
    let mut wrong = Vec::new();
    if field(m, "sent_by_user_id") != sender {
        wrong.push(format!("sent_by_user_id {:?}", field(m, "sent_by_user_id")));
    }
    if field(m, "magickspace_id") != space {
        wrong.push(format!("magickspace_id {:?}", field(m, "magickspace_id")));
    }
    if field(m, "content") != content {
        wrong.push("content changed".into());
    }
    if let Err(e) = expect_type(m, &["TEXT"]) {
        wrong.push(e);
    }
    if wrong.is_empty() {
        Ok("intact".into())
    } else {
        Err(wrong.join("; "))
    }
}

fn check_history(items: &[Value], want: &[(&str, &[&str], &str)]) -> Outcome {
    let wrong: Vec<String> = want
        .iter()
        .filter(|(id, _, _)| !id.is_empty())
        .filter_map(
            |(id, types, what)| match items.iter().find(|m| field(m, "id") == *id) {
                None => Some(format!("{what} is missing")),
                Some(m) => expect_type(m, types).err().map(|e| format!("{what} {e}")),
            },
        )
        .collect();
    if wrong.is_empty() {
        Ok("every message kept its type".into())
    } else {
        Err(wrong.join("; "))
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
