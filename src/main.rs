use anyhow::Result;
use cudarc::{cublas::CudaBlas, driver::CudaContext};
use tokenizers::Tokenizer;

use crate::models::Qwen2Model;

mod models;

fn main() -> Result<()> {
    let device = CudaContext::new(0)?;
    let blas = CudaBlas::new(device.default_stream())?;

    let model_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";

    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!(
        "Tokenizer loaded. Vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // 2. Load Model
    let model = Qwen2Model::load(&device, model_path)?;
    println!("Model loaded successfully.");

    // 3. Encode input
    let prompt = "Hello, AI!";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
    let input_ids = encoding.get_ids();
    println!("Prompt: '{}' -> IDs: {:?}", prompt, input_ids);

    // 4. Decode (Verify)
    let decoded = tokenizer
        .decode(input_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    println!("Decoded back: '{}'", decoded);

    // std::thread::sleep(std::time::Duration::from_secs(5));
    Ok(())
}
