use anyhow::Result;
use cudarc::cublas::CudaBlas;
use tokenizers::Tokenizer;

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

    // 3. Configure Sampler (Temp=0.7, Top-P=0.9)
    let mut sampler = Sampler::new(42, 0.7, 0.9, 0);

    // 4. Encode input
    let prompt = "请用一句话解释量子计算。";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
    let input_ids = encoding.get_ids();
    println!("Prompt: '{}' -> IDs: {:?}", prompt, input_ids);

    // 5. Generation
    let stream = device.default_stream();
    let mut cache_pos = 0;

    print!("{}", prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    // Prefill
    for &id in input_ids.iter().take(input_ids.len() - 1) {
        let _ = model.forward(&stream, &blas, &[id], cache_pos)?;
        cache_pos += 1;
    }

    let mut next_token_id = *input_ids.last().unwrap(); // u32

    for _ in 0..100 {
        model.forward(&stream, &blas, &[next_token_id], cache_pos)?;
        cache_pos += 1;

        let mut logits = model.sample(&device, &stream, &blas)?;

        // Advanced Sampling
        next_token_id = sampler.sample(&mut logits)?;

        let token = tokenizer.decode(&[next_token_id], true).unwrap();
        print!("{}", token);
        std::io::stdout().flush()?;

        // Stop tokens for Qwen
        if next_token_id == 151643 || next_token_id == 151645 {
            break;
        }
    }
    println!();

    Ok(())
}
