use anyhow::Result;
use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaContext;
use std::time::Instant;
use tokenizers::Tokenizer;

use lucciola::models::Qwen2Model;

fn main() -> Result<()> {
    let device = CudaContext::new(0)?;
    let blas = CudaBlas::new(device.default_stream())?;

    let model_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";

    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 2. Load Model
    println!("Loading model...");
    let mut model = Qwen2Model::load(&device, model_path)?;
    println!("Model loaded.");

    // 3. Prepare Input
    let prompt = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them.";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = encoding.get_ids();
    let n_input = input_ids.len();
    println!("Prompt length: {} tokens", n_input);

    let stream = device.default_stream();
    let mut cache_pos = 0;

    // --- Benchmarking Prefill (TTFT) ---
    println!("Benchmarking Prefill...");
    stream.synchronize()?;
    let start_prefill = Instant::now();

    // Process all but last
    for &id in input_ids.iter().take(n_input - 1) {
        let _ = model.forward(&stream, &blas, &[id], cache_pos)?;
        cache_pos += 1;
    }

    // Process last input token to get first output
    let last_input = *input_ids.last().unwrap();
    let hidden = model.forward(&stream, &blas, &[last_input], cache_pos)?;
    cache_pos += 1;
    let _logits = model.sample(&device, &stream, &blas, &hidden)?;

    stream.synchronize()?;
    let prefill_duration = start_prefill.elapsed();
    let ttft_ms = prefill_duration.as_secs_f64() * 1000.0;

    println!("Prefill done.");
    println!("Time to First Token (TTFT): {:.2} ms", ttft_ms);

    // --- Benchmarking Generation ---
    println!("Benchmarking Generation (50 tokens)...");
    let n_gen = 50;

    let mut next_token_id = last_input;

    stream.synchronize()?;
    let start_gen = Instant::now();

    for _ in 0..n_gen {
        let hidden = model.forward(&stream, &blas, &[next_token_id], cache_pos)?;
        cache_pos += 1;

        let logits = model.sample(&device, &stream, &blas, &hidden)?;

        let (id, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        next_token_id = id as u32;
    }

    stream.synchronize()?;
    let gen_duration = start_gen.elapsed();
    let tps = n_gen as f64 / gen_duration.as_secs_f64();
    let avg_latency = gen_duration.as_secs_f64() * 1000.0 / n_gen as f64;

    println!("Generation done.");
    println!("Throughput: {:.2} tokens/sec", tps);
    println!("Avg Latency per Token: {:.2} ms", avg_latency);

    Ok(())
}
