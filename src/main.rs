use anyhow::Result;
use cudarc::cublas::CudaBlas;
use tokenizers::Tokenizer;

use lucciola::chat::ChatTemplate;
use lucciola::models::Qwen2Model;
use lucciola::sampler::Sampler;

fn main() -> Result<()> {
    let device = cudarc::driver::CudaContext::new(0)?;
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
    let mut model = Qwen2Model::load(&device, model_path)?;
    println!("Model loaded successfully.");

    // 3. Configure Sampler (Temp=0.8, Top-P=0.9)
    let mut sampler = Sampler::new(42, 0.8, 0.9, 0);

    // 4. Chat Template
    let mut chat = ChatTemplate::new(Some(&model.config.model_type));
    chat.add("system", "You are a helpful assistant.");
    chat.add("user", "请用一句话解释量子计算。");

    let input_ids = chat.apply(&tokenizer)?;

    println!("Prompt: '{}'", "请用一句话解释量子计算。");

    // 5. Generation
    let stream = device.default_stream();
    let mut cache_pos = 0;

    print!("Response: ");
    use std::io::Write;
    std::io::stdout().flush()?;

    // Prefill (Batched)
    model.forward(&stream, &blas, &input_ids, cache_pos)?;
    cache_pos += input_ids.len();

    // Sample first token
    let mut logits = model.sample(&device, &stream, &blas)?;
    let mut next_token_id = sampler.sample(&mut logits)?;

    let token = tokenizer.decode(&[next_token_id], true).unwrap();
    print!("{}", token);
    std::io::stdout().flush()?;

    let eos_token_id = model.config.eos_token_id;
    let bos_token_id = model.config.bos_token_id;

    // Some models use BOS (like <|endoftext|>) as a stop token in certain contexts or as a fallback
    if next_token_id == eos_token_id || next_token_id == bos_token_id {
        println!();
        return Ok(());
    }

    for _ in 0..512 {
        model.forward(&stream, &blas, &[next_token_id], cache_pos)?;
        cache_pos += 1;

        let mut logits = model.sample(&device, &stream, &blas)?;
        next_token_id = sampler.sample(&mut logits)?;

        let token = tokenizer.decode(&[next_token_id], true).unwrap();
        print!("{}", token);
        std::io::stdout().flush()?;

        // Stop tokens
        if next_token_id == eos_token_id || next_token_id == bos_token_id {
            break;
        }
    }
    println!();

    Ok(())
}
