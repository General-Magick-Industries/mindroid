//! MagickMind Persona CLI: Centrifugo transport + LiteLLM inference + prepared persona.
//!
//! Identical to `magickmind_cli` but uses the bifrost `POST /v1/persona/{id}/prepare`
//! endpoint to fetch a ready-to-use system prompt per-request, with dyadic
//! (per-user) trait blending.
//!
//! Flow per message:
//!   1. ContextPreparer        — fetch chat history + knowledge from MagickMind
//!   2. Persona prepare        — fetch system prompt from bifrost persona endpoint
//!   3. CorpusClient           — query corpus for raw documents
//!   4. gpt-oss (distill)      — summarise raw docs into concise context
//!   5. SimpleContextBuilder   — assemble LLM messages (persona prompt + history + user)
//!   6. GenericLlmProcessor    — streaming inference via main model
//!   7. PostProcessor          — clean up response text
//!   8. MagickmindPersistence  — save response back to MagickMind
//!
//! Run with:
//!   cargo run -p mindroid-example-magickmind-persona-cli -- --config examples/magickmind_persona_cli/config.toml

use std::sync::Arc;

use mindroid::llm_client::{ChatRequest, LlmClient};
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, CorpusClient, GenericLlmProcessor, LlmMessage, MindroidConfig, Pipeline,
    PipelineContext, PostProcessor, Runtime, SimpleContextBuilder,
};

/// Response from `POST /v1/persona/{id}/prepare`.
#[derive(serde::Deserialize)]
struct PreparePersonaResponse {
    system_prompt: String,
}

/// Call the bifrost persona prepare endpoint.
///
/// `POST {base_url}/v1/persona/{persona_id}/prepare`
/// Body: `{ "user_id": "..." }` (optional)
///
/// Returns the assembled system prompt with effective personality traits.
async fn prepare_persona(
    http: &reqwest::Client,
    base_url: &str,
    persona_id: &str,
    user_id: Option<&str>,
    auth: &dyn mindroid::Auth,
) -> anyhow::Result<String> {
    let url = format!(
        "{}/v1/persona/{}/prepare",
        base_url.trim_end_matches('/'),
        persona_id
    );

    let headers = mindroid::auth::build_auth_header_map(auth).await?;

    let mut body = serde_json::Map::new();
    if let Some(uid) = user_id {
        body.insert(
            "user_id".to_string(),
            serde_json::Value::String(uid.to_string()),
        );
    }

    let resp = http
        .post(&url)
        .headers(headers)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Persona prepare failed ({status}): {text}");
    }

    let parsed: PreparePersonaResponse = resp.json().await?;
    Ok(parsed.system_prompt)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn,mindroid=info")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    // Resolve LLM client configs from [providers] + [models]
    let llm_config = config.llm("main")?;
    let distill_config = config.llm("distill").ok();

    // Auto-build transport (centrifugo), auth (apikey), memory (magickmind), observer from config
    let builder = Runtime::from_config(config)?;

    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();

    // Resolve MagickMind platform URL: prefer provider, fall back to auth.base_url
    let magickmind_url = if let Some(ref provider_name) = config.memory.provider {
        config
            .resolve_provider(provider_name, None)?
            .base_url
    } else {
        config
            .auth
            .base_url
            .as_deref()
            .unwrap_or("https://dev-magickmind.magickmind.ai")
            .to_string()
    };

    let mut magickmind_client = MagickmindClient::new(&magickmind_url, identity.clone());
    if let Some(key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(key);
    }
    let magickmind = Arc::new(magickmind_client);

    let agent_id = config.agent.agent_id.clone();

    // Persona config
    let persona_id = config
        .persona
        .persona_id
        .clone()
        .expect("persona.persona_id is required");

    let persona_url = if let Some(ref provider_name) = config.persona.provider {
        config
            .resolve_provider(provider_name, config.persona.base_url.as_deref())?
            .base_url
    } else {
        // Legacy fallback: persona.base_url → auth.base_url
        config
            .persona
            .base_url
            .as_deref()
            .or(config.auth.base_url.as_deref())
            .unwrap_or("https://dev-magickmind.magickmind.ai")
            .to_string()
    };

    tracing::info!("Persona enabled: id={persona_id}, url={persona_url}");

    let http = Arc::new(reqwest::Client::new());

    // Context preparer: fetch chat history and knowledge from MagickMind
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    // Corpus client: query documents from corpus (if configured)
    let corpus = if let Some(corpus_id) = &config.corpus.corpus_id {
        let corpus_url = if let Some(ref provider_name) = config.corpus.provider {
            config
                .resolve_provider(provider_name, config.corpus.base_url.as_deref())?
                .base_url
        } else {
            // Legacy fallback: corpus.base_url → auth.base_url
            config
                .corpus
                .base_url
                .as_deref()
                .or(config.auth.base_url.as_deref())
                .unwrap_or("https://dev-magickmind.magickmind.ai")
                .to_string()
        };

        let mut client = CorpusClient::new(&corpus_url, identity.clone());
        if let Some(key) = config.corpus.api_key.as_ref().or(config.auth.api_key.as_ref()) {
            client = client.with_api_key(key);
        }

        tracing::info!("Corpus RAG enabled for corpus_id={corpus_id}");
        Some((Arc::new(client), corpus_id.clone()))
    } else {
        None
    };

    // Distillation LLM: summarise raw corpus docs before adding to context
    let distill_llm = distill_config.map(|cfg| {
        tracing::info!("Corpus distillation enabled via [models.distill]");
        Arc::new(LlmClient::new(cfg).expect("Failed to create distillation LLM client"))
    });

    let llm_config = Arc::new(llm_config);

    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let magickmind = Arc::clone(&magickmind);
            let llm_config = Arc::clone(&llm_config);
            let corpus = corpus.clone();
            let distill_llm = distill_llm.clone();
            let http = Arc::clone(&http);
            let identity = Arc::clone(&identity);
            let persona_id = persona_id.clone();
            let persona_url = persona_url.clone();

            async move {
                // Step 1: fetch conversation context from MagickMind
                let mut context = match preparer.prepare(&ctx.message).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(
                            "MagickMind context fetch failed, continuing without history: {e}"
                        );
                        Vec::new()
                    }
                };

                // Step 2: fetch the prepared system prompt from bifrost persona endpoint
                let user_id = ctx.message.sender_id.clone();
                let system_prompt = match prepare_persona(
                    &http,
                    &persona_url,
                    &persona_id,
                    Some(&user_id),
                    identity.as_ref(),
                )
                .await
                {
                    Ok(prompt) => {
                        tracing::info!(
                            "Persona prepared: {} bytes for user={}",
                            prompt.len(),
                            user_id
                        );
                        prompt
                    }
                    Err(e) => {
                        tracing::error!("Persona prepare failed: {e}");
                        return;
                    }
                };

                // Step 3: classify whether corpus retrieval is needed
                let needs_corpus = if corpus.is_some() {
                    if let Some(ref llm) = distill_llm {
                        let classify_messages = vec![
                            LlmMessage::system(
                                "You are a message classifier. Determine if the user's message \
                                 requires looking up reference documents to answer properly.\n\n\
                                 Reply with ONLY \"yes\" or \"no\".\n\n\
                                 Say \"no\" for: greetings, small talk, thank you, yes/no answers, \
                                 and simple conversational messages.\n\
                                 Say \"yes\" for: questions about features, APIs, configuration, \
                                 troubleshooting, how-to requests, or anything that needs factual \
                                 knowledge to answer.",
                            ),
                            LlmMessage::user(&ctx.message.content),
                        ];

                        match llm
                            .chat(ChatRequest {
                                messages: &classify_messages,
                                model: None,
                                temperature: Some(0.0),
                                max_tokens: Some(3),
                                stream: false,
                                response_format: None,
                            })
                            .await
                        {
                            Ok((answer, _)) => {
                                let needs = answer.trim().to_lowercase().contains("yes");
                                tracing::info!(
                                    "Corpus gate: \"{}\" → {}",
                                    ctx.message.content,
                                    if needs { "QUERY" } else { "SKIP" }
                                );
                                needs
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Corpus gate classification failed, querying anyway: {e}"
                                );
                                true
                            }
                        }
                    } else {
                        // No distill LLM configured, always query
                        true
                    }
                } else {
                    false
                };

                // Step 4: query corpus and distill if needed
                if needs_corpus {
                    if let Some((ref corpus_client, ref corpus_id)) = corpus {
                        match corpus_client
                            .query(corpus_id, &ctx.message.content, None)
                            .await
                        {
                            Ok(raw) if !raw.is_empty() => {
                                let corpus_context = if let Some(ref llm) = distill_llm {
                                    let distill_messages = vec![
                                        LlmMessage::system(
                                            "You are a context distillation assistant. Given a user \
                                             question and retrieved documents, extract only the \
                                             information relevant to answering the question. Be concise \
                                             and preserve key facts, names, and numbers. Do not answer \
                                             the question — only summarise the relevant context.",
                                        ),
                                        LlmMessage::user(format!(
                                            "User question: {}\n\nDocuments:\n{}",
                                            ctx.message.content, raw
                                        )),
                                    ];

                                    match llm
                                        .chat(ChatRequest {
                                            messages: &distill_messages,
                                            model: None,
                                            temperature: Some(0.0),
                                            max_tokens: None,
                                            stream: false,
                                            response_format: None,
                                        })
                                        .await
                                    {
                                        Ok((summary, usage)) => {
                                            tracing::info!(
                                                "Corpus distilled: {} → {} bytes (tokens: {:?})",
                                                raw.len(),
                                                summary.len(),
                                                usage
                                            );
                                            summary
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "Corpus distillation failed, using raw: {e}"
                                            );
                                            raw
                                        }
                                    }
                                } else {
                                    raw
                                };

                                context.push(LlmMessage::system(format!(
                                    "Reference documents:\n{corpus_context}"
                                )));
                            }
                            Ok(_) => {
                                tracing::debug!("Corpus returned empty result");
                            }
                            Err(e) => {
                                tracing::warn!("Corpus query failed, continuing without: {e}");
                            }
                        }
                    }
                }

                // Step 5: build LLM client for this request
                let llm_client = match LlmClient::new((*llm_config).clone()) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("LLM client error: {e}");
                        return;
                    }
                };

                // Step 6: assemble pipeline with prepared persona prompt + context
                let pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        &system_prompt,
                        Arc::new(context),
                    ))
                    .add_streaming_stage(GenericLlmProcessor::new(llm_client))
                    .add_stage(PostProcessor)
                    .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                match ctx.run_with_context(&pipeline, &mut pctx).await {
                    Ok(None) => {
                        tracing::info!("No response generated");
                        return;
                    }
                    Ok(Some(_)) => {}
                    Err(e) => {
                        tracing::error!("Pipeline error: {e}");
                        return;
                    }
                }

                let response = pctx.response.as_deref().unwrap_or("").trim().to_string();
                if !response.is_empty()
                    && let Err(e) = ctx.respond(&response).await
                {
                    tracing::error!("Send error: {e}");
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}
