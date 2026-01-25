use axum::response::sse::Event;
use axum::{
    Json, Router,
    extract::State,
    response::{IntoResponse, Sse},
    routing::post,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::{Any, CorsLayer};

use lucciola::models::Qwen2Model; // Logic is compatible with DeepSeek if loaded correctly
use lucciola::sampler::Sampler;

// --- Request/Response Structures ---

#[derive(Deserialize, Debug)]
struct CompletionRequest {
    prompt: String,
    suffix: Option<String>, // Key for FIM
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>, // Stop sequences
    stream: Option<bool>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}

#[derive(Serialize)]
struct CompletionChunkChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

struct AppState {
    model: Mutex<Qwen2Model>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
    // Pre-computed Special Token IDs
    fim_begin_id: Option<u32>,
    fim_hole_id: Option<u32>,
    fim_end_id: Option<u32>,
}

// --- Main ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Try to find the model. Prioritize deepseek-coder if available.
    // let possible_paths = [
    //     "/app/lucciola/models/deepseek-coder-1.3b-base",
    //     "/app/lucciola/models/deepseek-coder-6.7b-base",
    //     "/app/lucciola/models/Qwen2.5-0.5B-Instruct",
    // ];

    // let model_path = possible_paths.iter()
    //     .find(|p| std::path::Path::new(p).exists())
    //     .ok_or_else(|| anyhow::anyhow!("No model found. Please download deepseek-coder-1.3b-base."))?;
    // let model_path = "/app/lucciola/models/deepseek-coder-1.3b-base";
    let model_path = "/app/lucciola/models/deepseek-coder-6.7b-base";

    // println!("Loading model from {}...", model_path);
    println!("Loading model...");
    let model = Qwen2Model::load(0, model_path)?;
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Look up FIM tokens
    let fim_begin = tokenizer.token_to_id("<｜fim▁begin｜>");
    let fim_hole = tokenizer.token_to_id("<｜fim▁hole｜>");
    let fim_end = tokenizer.token_to_id("<｜fim▁end｜>");

    if let (Some(b), Some(h), Some(e)) = (fim_begin, fim_hole, fim_end) {
        println!("FIM Tokens Detected: Begin={}, Hole={}, End={}", b, h, e);
    } else {
        println!("Warning: FIM tokens not found in tokenizer. FIM might not work correctly.");
    }

    println!("Model loaded successfully. Starting Completion Server...");

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        model_name: model_path.split('/').last().unwrap().to_string(),
        fim_begin_id: fim_begin,
        fim_hole_id: fim_hole,
        fim_end_id: fim_end,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/completions", post(completions_handler))
        .layer(cors)
        .with_state(state);

    let address_str = "0.0.0.0:3000";
    let addr: SocketAddr = address_str.parse()?;
    println!("Listening on http://{}", address_str);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// --- Handler ---

async fn completions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    println!(
        "Request: FIM={:?} Stream={:?}",
        request.suffix.is_some(),
        request.stream
    );

    // 1. Construct Input IDs
    // Case A: FIM (Prompt + Suffix)
    // Case B: Normal (Prompt only)

    let mut input_ids = Vec::new();

    // 0. Add BOS (Beginning of Sequence) if available.
    // DeepSeek/Llama models perform better with explicit BOS.
    // DeepSeek-Coder config usually has bos_token_id = 32013.
    let bos_id = {
        let guard = state.model.lock().unwrap();
        guard.config.bos_token_id
    };
    input_ids.push(bos_id);

    if let Some(suffix) = &request.suffix {
        // FIM Mode: <begin> PROMPT <hole> SUFFIX <end>
        if let (Some(b), Some(h), Some(e)) =
            (state.fim_begin_id, state.fim_hole_id, state.fim_end_id)
        {
            input_ids.push(b);
            let prompt_ids = state
                .tokenizer
                .encode(request.prompt.clone(), false)
                .unwrap();
            input_ids.extend(prompt_ids.get_ids());

            input_ids.push(h);
            let suffix_ids = state.tokenizer.encode(suffix.clone(), false).unwrap();
            input_ids.extend(suffix_ids.get_ids());

            input_ids.push(e);
        } else {
            // Fallback: Just concatenate? Or fail? Better to just use prompt.
            eprintln!("FIM requested but tokens missing. Falling back to prefix-only.");
            let ids = state
                .tokenizer
                .encode(request.prompt.clone(), false)
                .unwrap();
            input_ids.extend(ids.get_ids());
        }
    } else {
        // Normal Mode or Client-Formatted FIM
        // Check if the prompt contains StarCoder/CodeLlama style text markers
        // Format: <fim_prefix> PREFIX <fim_suffix> SUFFIX <fim_middle>
        if request.prompt.contains("<fim_prefix>")
            && request.prompt.contains("<fim_suffix>")
            && request.prompt.contains("<fim_middle>")
        {
            // Detected client-side FIM formatting
            println!("DEBUG: Detected client-side FIM markers in prompt string.");
            if let (Some(b), Some(h), Some(e)) =
                (state.fim_begin_id, state.fim_hole_id, state.fim_end_id)
            {
                // Extract valid parts
                // String looks like: "<fim_prefix>...<fim_suffix>...<fim_middle>"
                // We split by <fim_suffix> first
                let parts: Vec<&str> = request.prompt.split("<fim_suffix>").collect();
                if parts.len() >= 2 {
                    // Part 0 contains <fim_prefix> and the actual prefix
                    let raw_prefix = parts[0];
                    // Part 1 contains the actual suffix and <fim_middle>
                    let raw_suffix = parts[1];

                    let prefix = raw_prefix.replace("<fim_prefix>", "");
                    let suffix = raw_suffix.replace("<fim_middle>", "");

                    // 1. Begin
                    input_ids.push(b);
                    // 2. Prefix
                    input_ids.extend(state.tokenizer.encode(prefix, false).unwrap().get_ids());
                    // 3. Hole
                    input_ids.push(h);
                    // 4. Suffix
                    input_ids.extend(state.tokenizer.encode(suffix, false).unwrap().get_ids());
                    // 5. End
                    input_ids.push(e);
                } else {
                    // Malformed split
                    let ids = state
                        .tokenizer
                        .encode(request.prompt.clone(), false)
                        .unwrap();
                    input_ids.extend(ids.get_ids());
                }
            } else {
                let ids = state
                    .tokenizer
                    .encode(request.prompt.clone(), false)
                    .unwrap();
                input_ids.extend(ids.get_ids());
            }
        } else {
            // Truly Normal Mode
            // We manually added BOS above, so encode with false to avoid double BOS
            let ids = state
                .tokenizer
                .encode(request.prompt.clone(), false)
                .unwrap();
            input_ids.extend(ids.get_ids());
        }
    }

    println!(
        "Request: FIM={} Stream={:?} InputLen={} First10={:?}",
        request.suffix.is_some(),
        request.stream,
        input_ids.len(),
        input_ids.iter().take(10).collect::<Vec<_>>()
    );

    println!(
        "DEBUG INPUT Prompt: {:?}, Suffix: {:?}",
        request.prompt, request.suffix
    );
    let decoded_input = state
        .tokenizer
        .decode(&input_ids, false)
        .unwrap_or_default();
    println!("DEBUG FULL INPUT DECODED: {:?}", decoded_input);

    let is_stream = request.stream.unwrap_or(false);
    let model_name = state.model_name.clone();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let id = format!(
        "cmpl-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    // Stop sequences handling
    let stop_sequences = request.stop.clone().unwrap_or_default();

    if is_stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, axum::Error>>();
        let state_model = state.clone();

        tokio::task::spawn_blocking(move || {
            if input_ids.is_empty() {
                return;
            }

            let mut model = state_model.model.lock().unwrap();
            let mut sampler = Sampler::new(
                None,
                request.temperature.unwrap_or(0.1), // Coder usually low temp
                request.top_p.unwrap_or(0.95),
                1024,
            );
            let max_tokens = request.max_tokens.unwrap_or(128); // Default small for completion

            let mut current_text = String::new();

            let _ = model.generate(
                &input_ids,
                &mut sampler,
                max_tokens,
                &state_model.tokenizer,
                |token| {
                    current_text.push_str(token);

                    // Stop Check
                    for stop_seq in &stop_sequences {
                        if current_text.contains(stop_seq) {
                            // If we hit a stop sequence, we shouldn't send the full token if it contains the stop seq part.
                            // But for simplicity in this stream, we just stop *after* sending.
                            // Better: check if `token` completes a stop seq.
                            // This naive check might send the stop token. That's acceptable for now.
                            return false;
                        }
                    }

                    let chunk = CompletionChunk {
                        id: id.clone(),
                        object: "text_completion".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: token.to_string(),
                            index: 0,
                            finish_reason: None,
                        }],
                    };

                    if let Ok(_) = tx.send(Ok(Event::default().json_data(chunk).unwrap())) {
                        true
                    } else {
                        false
                    }
                },
            );

            // Finish
            println!("DEBUG GENERATED TEXT: {:?}", current_text);
            let _ = tx.send(Ok(Event::default()
                .json_data(CompletionChunk {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: "".to_string(),
                        index: 0,
                        finish_reason: Some("stop".to_string()),
                    }],
                })
                .unwrap()));
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        });

        let stream = UnboundedReceiverStream::new(rx);
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        // Non-streaming
        let generated_text = if !input_ids.is_empty() {
            let state_model = state.clone();
            tokio::task::spawn_blocking(move || {
                let mut model = state_model.model.lock().unwrap();
                let mut text_buffer = String::new();
                let mut sampler = Sampler::new(
                    None,
                    request.temperature.unwrap_or(0.1),
                    request.top_p.unwrap_or(0.95),
                    1024,
                );

                let _ = model.generate(
                    &input_ids,
                    &mut sampler,
                    request.max_tokens.unwrap_or(128),
                    &state_model.tokenizer,
                    |token| {
                        text_buffer.push_str(token);
                        for stop_seq in &stop_sequences {
                            if text_buffer.contains(stop_seq) {
                                // Basic truncation
                                if let Some(idx) = text_buffer.find(stop_seq) {
                                    text_buffer.truncate(idx);
                                }
                                return false;
                            }
                        }
                        true
                    },
                );
                text_buffer
            })
            .await
            .unwrap_or_default()
        } else {
            String::new()
        };

        Json(CompletionResponse {
            id,
            object: "text_completion".to_string(),
            created,
            model: model_name,
            choices: vec![CompletionChoice {
                text: generated_text,
                index: 0,
                finish_reason: "stop".to_string(),
            }],
        })
        .into_response()
    }
}
