use tower_http::cors::{CorsLayer, Any};
use axum::{
    extract::State,
    routing::post,
    Json, Router,
    response::{IntoResponse, Sse},
};
use axum::response::sse::Event;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use futures::stream::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use lucciola::models::Qwen2Model;
use lucciola::sampler::Sampler;
use lucciola::chat::ChatTemplate;

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
}

// Non-streaming response structs
#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ChatMessageResponse,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMessageResponse {
    role: String,
    content: String,
}

// Streaming response structs
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChunkDelta {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check if models exist
    let model_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";
    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!("Model not found at {}. Please download it first.", model_path);
    }
    
    println!("Loading model from {}...", model_path);
    let model = Qwen2Model::load(0, model_path)?;
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    println!("Model loaded successfully. Starting API server...");

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        model_name: "qwen2.5-0.5b".to_string(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(cors)
        .with_state(state);

    let address_str = "0.0.0.0:3000";
    let addr: SocketAddr = address_str.parse()?;
    println!("Listening on http://{}", address_str);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    println!("Received request: {} messages, stream={:?}", request.messages.len(), request.stream);

    let mut template = ChatTemplate::new(Some("qwen2"));
    for msg in &request.messages {
        template.add(&msg.role, &msg.content);
    }

    let input_ids = match template.apply(&state.tokenizer) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("Error applying chat template: {}", e);
            return Json(serde_json::json!({"error": e.to_string()})).into_response();
        }
    };
    
    println!("Generated {} Input IDs", input_ids.len());

    let is_stream = request.stream.unwrap_or(false);
    let model_name = state.model_name.clone();
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let id = format!("chatcmpl-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos());

    if is_stream {
        // --- Streaming Response ---
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, axum::Error>>();
        let state_model = state.clone();
        
        // Spawn blocking task for inference
        tokio::task::spawn_blocking(move || {
            if input_ids.is_empty() { return; }

            let mut model = state_model.model.lock().unwrap();
            let mut sampler = Sampler::new(
                None, 
                request.temperature.unwrap_or(0.7), 
                request.top_p.unwrap_or(0.9), 
                1024
            );
            let max_tokens = request.max_tokens.unwrap_or(512);
            
            println!("Starting streaming generation...");
            
            // 1. Send Role Chunk
            let _ = tx.send(Ok(Event::default().json_data(ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta { role: Some("assistant".to_string()), content: None },
                    finish_reason: None,
                }],
            }).unwrap()));

            // 2. Generate and Send Content Chunks
            let _ = model.generate(
                &input_ids,
                &mut sampler,
                max_tokens,
                &state_model.tokenizer,
                |token| {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta { role: None, content: Some(token.to_string()) },
                            finish_reason: None,
                        }],
                    };
                    
                    if let Ok(_) = tx.send(Ok(Event::default().json_data(chunk).unwrap())) {
                        true // continue
                    } else {
                        false // channel closed, stop generation
                    }
                }
            );

            // 3. Send Finish Chunk
            let _ = tx.send(Ok(Event::default().json_data(ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta { role: None, content: None },
                    finish_reason: Some("stop".to_string()),
                }],
            }).unwrap()));
            
            // 4. Send [DONE]
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
            
            println!("Streaming complete.");
        });

        // Return SSE Stream
        let stream = UnboundedReceiverStream::new(rx);
        Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()).into_response()

    } else {
        // --- Non-Streaming Response ---
        let generated_text = if !input_ids.is_empty() {
            let state_model = state.clone();
            tokio::task::spawn_blocking(move || {
                let mut model = state_model.model.lock().unwrap();
                let mut text_buffer = String::new();
                let mut sampler = Sampler::new(
                    None, 
                    request.temperature.unwrap_or(0.7), 
                    request.top_p.unwrap_or(0.9), 
                    1024
                );
                let _ = model.generate(
                    &input_ids,
                    &mut sampler,
                    request.max_tokens.unwrap_or(512),
                    &state_model.tokenizer,
                    |token| {
                        text_buffer.push_str(token);
                        true 
                    }
                );
                text_buffer
            }).await.unwrap_or_default()
        } else {
            String::new()
        };

        Json(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: "stop".to_string(),
            }],
        }).into_response()
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    println!("Received completion request: prompt len={}, stream={:?}", request.prompt.len(), request.stream);

    // For raw completions, we don't use ChatTemplate. We encode the prompt directly.
    let input_ids = match state.tokenizer.encode(request.prompt.clone(), true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            eprintln!("Error encoding prompt: {}", e);
            return Json(serde_json::json!({"error": e.to_string()})).into_response();
        }
    };

    println!("Generated {} Input IDs", input_ids.len());

    let is_stream = request.stream.unwrap_or(false);
    let model_name = state.model_name.clone();
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let id = format!("cmpl-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos());

    if is_stream {
        // --- Streaming Response ---
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, axum::Error>>();
        let state_model = state.clone();
        
        tokio::task::spawn_blocking(move || {
            if input_ids.is_empty() { return; }

            let mut model = state_model.model.lock().unwrap();
            let mut sampler = Sampler::new(
                None, 
                request.temperature.unwrap_or(0.7), 
                request.top_p.unwrap_or(0.9), 
                1024
            );
            let max_tokens = request.max_tokens.unwrap_or(512);
            
            println!("Starting streaming completion...");
            
            let _ = model.generate(
                &input_ids,
                &mut sampler,
                max_tokens,
                &state_model.tokenizer,
                |token| {
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
                }
            );

            // Send Finish Chunk
            let _ = tx.send(Ok(Event::default().json_data(CompletionChunk {
                id: id.clone(),
                object: "text_completion".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![CompletionChunkChoice {
                    text: "".to_string(),
                    index: 0,
                    finish_reason: Some("stop".to_string()),
                }],
            }).unwrap()));
            
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
            
            println!("Streaming completion complete.");
        });

        let stream = UnboundedReceiverStream::new(rx);
        Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()).into_response()

    } else {
        // --- Non-Streaming Response ---
        let generated_text = if !input_ids.is_empty() {
            let state_model = state.clone();
            tokio::task::spawn_blocking(move || {
                let mut model = state_model.model.lock().unwrap();
                let mut text_buffer = String::new();
                let mut sampler = Sampler::new(
                    None, 
                    request.temperature.unwrap_or(0.7), 
                    request.top_p.unwrap_or(0.9), 
                    1024
                );
                let _ = model.generate(
                    &input_ids,
                    &mut sampler,
                    request.max_tokens.unwrap_or(512),
                    &state_model.tokenizer,
                    |token| {
                        text_buffer.push_str(token);
                        true 
                    }
                );
                text_buffer
            }).await.unwrap_or_default()
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
        }).into_response()
    }
}
