use anyhow::Result;
use std::time::Instant;
use tokenizers::Tokenizer;

use lucciola::models::Qwen2Model;

fn main() -> Result<()> {
    let model_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";

    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 2. Load Model
    println!("Loading model...");
    let mut model = Qwen2Model::load(0, model_path)?;
    println!("Model loaded.");

    // 3. Prepare Input
    let prompt = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them.";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = encoding.get_ids();
    let n_input = input_ids.len();
    println!("Prompt length: {} tokens", n_input);

    let stream = model.device.default_stream();
    let mut cache_pos = 0;

    // --- Benchmarking Prefill (TTFT) ---
    println!("Benchmarking Prefill...");
    stream.synchronize()?;
    let start_prefill = Instant::now();

    // Batched Prefill
    model.forward(input_ids, cache_pos)?;
    cache_pos += n_input;

    let _logits = model.sample()?;

    stream.synchronize()?;
    let prefill_duration = start_prefill.elapsed();
    let ttft_ms = prefill_duration.as_secs_f64() * 1000.0;

    println!("Prefill done.");
    println!("Time to First Token (TTFT): {:.2} ms", ttft_ms);

    // --- Benchmarking Generation ---
    println!("Benchmarking Generation (50 tokens)...");
    let n_gen = 200;

    // Get last token from input to start generation (though we verify sample next)
    // Actually we pick the token from the sample() result usually, but here for simple bench
    // we can just pick the argmax from the previous sample.
    let logits = model.sample()?;
    let mut next_token_id = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    stream.synchronize()?;
    let start_gen = Instant::now();

    for _ in 0..n_gen {
        model.forward(&[next_token_id], cache_pos)?;
        cache_pos += 1;

        let logits = model.sample()?;

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
