//! Corpus access probe: proves an account can reach one knowledge corpus end
//! to end — account login, end-user token mint, then a `CorpusTool` query
//! exactly as a spawner-run agent would make it.
//!
//! Run:
//!   MM_EMAIL=you@example.com MM_PASSWORD=... \
//!     cargo run -p mindroid-example-corpus-probe --bin corpus_probe -- <corpus_id> [query]
//!
//! Env: MM_BASE_URL (default https://dev-bifrost.magickmind.ai),
//! MM_API_KEY (optional LiteLLM key forwarded as x-api-key to fund retrieval).

use std::sync::Arc;

use anyhow::{Context as _, bail};
use mindroid::Auth;
use mindroid::auth::apikey::ApiKeyAuth;
use mindroid::auth::static_id::StaticAuth;
use mindroid::models::CredentialKind;
use mindroid::tools::{AgentCredentials, CorpusCatalog, CorpusTool, Tool, ToolContext};
use serde::Deserialize;
use serde_json::json;

const DEFAULT_BASE_URL: &str = "https://dev-bifrost.magickmind.ai";
const DEFAULT_QUERY: &str = "What is this knowledge base about? Summarize its contents.";

/// The account's default chat identity, provisioned by the website's first-run
/// pipeline — preferred because it always exists on a provisioned account.
const DEFAULT_CHAT_USER_MARKER: &str = "mm-default-chat-user";

#[derive(Deserialize)]
struct EndUserPage {
    #[serde(default)]
    data: Vec<EndUser>,
}

#[derive(Deserialize)]
struct EndUser {
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    external_id: Option<String>,
    #[serde(default)]
    participant_type: Option<String>,
}

#[derive(Deserialize)]
struct MintedToken {
    token: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let mut args = std::env::args().skip(1);
    let Some(corpus_id) = args.next() else {
        bail!("usage: corpus_probe <corpus_id> [query]");
    };
    let query = args.next().unwrap_or_else(|| DEFAULT_QUERY.to_string());

    let base_url = std::env::var("MM_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let email = std::env::var("MM_EMAIL").context("MM_EMAIL is required")?;
    let password = std::env::var("MM_PASSWORD").context("MM_PASSWORD is required")?;
    let api_key = std::env::var("MM_API_KEY").ok();

    // 1. Account login — the tenant credential everything else derives from.
    let account = ApiKeyAuth::new(&base_url, &email, &password);
    let jwt = account.get_token().await.context("account login failed")?;
    println!("[1/4] logged in as {email}");

    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    // 2. Pick a HUMAN end-user identity to query as (agents' credentials
    //    belong to their spawner processes alone).
    let page: EndUserPage = http
        .get(format!("{base_url}/v1/end-users"))
        .bearer_auth(&jwt)
        .send()
        .await?
        .error_for_status()
        .context("listing end users failed")?
        .json()
        .await?;
    let subject = page
        .data
        .iter()
        .find(|u| u.external_id.as_deref() == Some(DEFAULT_CHAT_USER_MARKER))
        .or_else(|| {
            page.data
                .iter()
                .find(|u| u.participant_type.as_deref() != Some("AGENT"))
        })
        .context("the account has no non-agent end user to query as")?;
    println!(
        "[2/4] querying as end user {:?} ({})",
        subject.name, subject.id
    );

    // 3. Mint a short-lived end-user token for that identity.
    let minted: MintedToken = http
        .post(format!("{base_url}/v1/end-users/tokens"))
        .bearer_auth(&jwt)
        .json(&json!({
            "subject_id": subject.id,
            "supervised": true,
            "ttl_seconds": 600,
        }))
        .send()
        .await?
        .error_for_status()
        .context("minting the end-user token failed")?
        .json()
        .await?;
    println!("[3/4] minted an end-user token");

    // 4. Query the corpus through CorpusTool, exactly as an agent turn would:
    //    the id rides as an activation grant, the catalog is empty (no space).
    let creds = AgentCredentials {
        agent_id: subject.id.clone(),
        auth: Arc::new(StaticAuth::new(minted.token)),
        credential_kind: CredentialKind::EndUser,
    };
    let ctx = ToolContext::default();
    ctx.set(creds);
    ctx.set(CorpusCatalog(Vec::new()));

    let tool = CorpusTool::new(&base_url, api_key).with_activation_ids(vec![corpus_id.clone()]);
    let out = tool
        .execute(json!({ "corpus_id": corpus_id, "query": query }), &ctx)
        .await
        .context("corpus query failed")?;

    println!("[4/4] corpus {corpus_id} answered:\n\n{out}");
    Ok(())
}
