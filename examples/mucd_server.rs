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

use lucciola::models::Qwen2Model;
use lucciola::mucd::MucdDecoder;

// ==================== 请求/响应结构体 ====================

#[derive(Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[allow(dead_code)]
    temperature: Option<f32>,
    #[allow(dead_code)]
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
}

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
    #[allow(dead_code)]
    temperature: Option<f32>,
    #[allow(dead_code)]
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

// ==================== 应用状态 ====================

struct AppState {
    decoder: Mutex<MucdDecoder>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

// ==================== main ====================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let main_model_path = "models/deepseek-coder-6.7b-base";
    let aux_model_path = "models/deepseek-coder-1.3b-base";

    // 检查模型是否存在
    if !std::path::Path::new(main_model_path).exists() {
        anyhow::bail!(
            "主模型未找到: {}。请先下载模型。",
            main_model_path
        );
    }
    if !std::path::Path::new(aux_model_path).exists() {
        anyhow::bail!(
            "辅助模型未找到: {}。请先下载模型。",
            aux_model_path
        );
    }

    // 加载 Tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", main_model_path))
        .map_err(|e| anyhow::anyhow!("无法加载 tokenizer: {}", e))?;
    println!(
        "Tokenizer 加载完成。Vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // 加载主模型
    println!("加载主模型: {} ...", main_model_path);
    let main_model = Qwen2Model::load(0, main_model_path, 0.7)?;
    println!("主模型加载完成。");

    // 加载辅助模型
    println!("加载辅助模型: {} ...", aux_model_path);
    let aux_model = Qwen2Model::load(0, aux_model_path, 0.9)?;
    println!("辅助模型加载完成。");

    // 创建 MUCD 解码器
    let decoder = MucdDecoder::new(main_model, aux_model, 0.1);
    println!("MUCD 解码器已初始化。");

    let state = Arc::new(AppState {
        decoder: Mutex::new(decoder),
        tokenizer,
        model_name: "deepseek-coder-6.7b-mucd".to_string(),
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
    println!("MUCD API 服务器启动: http://{}", address_str);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ==================== Chat Completions ====================

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    println!(
        "收到 chat 请求: {} 条消息, stream={:?}",
        request.messages.len(),
        request.stream
    );

    // 将所有消息内容拼接为 prompt（base 模型没有 chat 模板）
    let prompt: String = request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let input_ids = match state.tokenizer.encode(prompt, true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            eprintln!("编码错误: {}", e);
            return Json(serde_json::json!({"error": e.to_string()})).into_response();
        }
    };

    println!("Input IDs 数量: {}", input_ids.len());

    let is_stream = request.stream.unwrap_or(false);
    let model_name = state.model_name.clone();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let id = format!(
        "chatcmpl-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    if is_stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, axum::Error>>();
        let state_ref = state.clone();

        tokio::task::spawn_blocking(move || {
            if input_ids.is_empty() {
                return;
            }

            let mut decoder = state_ref.decoder.lock().unwrap();
            decoder.reset();
            let max_tokens = request.max_tokens.unwrap_or(512);

            // 发送 role chunk
            let _ = tx.send(Ok(Event::default()
                .json_data(ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap()));

            let _ = decoder.generate(
                &input_ids,
                max_tokens,
                &state_ref.tokenizer,
                |token, _debug_info| {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(token.to_string()),
                            },
                            finish_reason: None,
                        }],
                    };

                    tx.send(Ok(Event::default().json_data(chunk).unwrap()))
                        .is_ok()
                },
            );

            // 发送结束 chunk
            let _ = tx.send(Ok(Event::default()
                .json_data(ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                })
                .unwrap()));

            let _ = tx.send(Ok(Event::default().data("[DONE]")));
            println!("Chat streaming 完成。");
        });

        let stream = UnboundedReceiverStream::new(rx);
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        let generated_text = if !input_ids.is_empty() {
            let state_ref = state.clone();
            tokio::task::spawn_blocking(move || {
                let mut decoder = state_ref.decoder.lock().unwrap();
                decoder.reset();
                let mut text_buffer = String::new();
                let _ = decoder.generate(
                    &input_ids,
                    request.max_tokens.unwrap_or(512),
                    &state_ref.tokenizer,
                    |token, _debug_info| {
                        text_buffer.push_str(token);
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
        })
        .into_response()
    }
}

// ==================== Completions ====================

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    println!(
        "收到 completion 请求: prompt len={}, stream={:?}",
        request.prompt.len(),
        request.stream
    );

    let input_ids = match state.tokenizer.encode(request.prompt.clone(), true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            eprintln!("编码错误: {}", e);
            return Json(serde_json::json!({"error": e.to_string()})).into_response();
        }
    };

    println!("Input IDs 数量: {}", input_ids.len());

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

    if is_stream {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Event, axum::Error>>();
        let state_ref = state.clone();

        tokio::task::spawn_blocking(move || {
            if input_ids.is_empty() {
                return;
            }

            let mut decoder = state_ref.decoder.lock().unwrap();
            decoder.reset();
            let max_tokens = request.max_tokens.unwrap_or(512);

            println!("开始 MUCD streaming completion...");

            let _ = decoder.generate(
                &input_ids,
                max_tokens,
                &state_ref.tokenizer,
                |token, _debug_info| {
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

                    tx.send(Ok(Event::default().json_data(chunk).unwrap()))
                        .is_ok()
                },
            );

            // 发送结束 chunk
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
            println!("MUCD streaming completion 完成。");
        });

        let stream = UnboundedReceiverStream::new(rx);
        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response()
    } else {
        let generated_text = if !input_ids.is_empty() {
            let state_ref = state.clone();
            tokio::task::spawn_blocking(move || {
                let mut decoder = state_ref.decoder.lock().unwrap();
                decoder.reset();
                let mut text_buffer = String::new();
                let _ = decoder.generate(
                    &input_ids,
                    request.max_tokens.unwrap_or(512),
                    &state_ref.tokenizer,
                    |token, _debug_info| {
                        text_buffer.push_str(token);
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
