use anyhow::Result;
use tokenizers::Tokenizer;

use lucciola::chat::ChatTemplate;
use lucciola::models::Qwen2Model;
use lucciola::sampler::Sampler;
use lucciola::streamer::Streamer;

fn main() -> Result<()> {
    let model_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";

    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!(
        "Tokenizer loaded. Vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // 2. Load Model
    let mut model = Qwen2Model::load(0, model_path)?;
    println!("Model loaded successfully.");

    // 3. Configure Sampler (Temp=0.8, Top-P=0.9)
    let mut sampler = Sampler::new(42, 0.8, 0.9, 0);

    let prompt = "请介绍一下量子计算的基本原理。";
    // 4. Chat Template
    let mut chat = ChatTemplate::new(Some(&model.config.model_type));
    chat.add("system", "You are a helpful assistant.");
    chat.add("user", prompt);

    let input_ids = chat.apply(&tokenizer)?;

    println!("Prompt: '{}'", prompt);

    // 5. Generation
    print!("Response: ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut streamer = Streamer::new(&tokenizer);

    model.generate(&input_ids, &mut sampler, 512, |token_id| {
        if let Some(text) = streamer.put(token_id) {
            print!("{}", text);
            let _ = std::io::stdout().flush();
        }
        true // continue
    })?;
    println!();

    Ok(())
}
