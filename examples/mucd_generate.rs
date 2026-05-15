use anyhow::Result;
use tokenizers::Tokenizer;

use lucciola::models::Qwen2Model;
use lucciola::mucd::MucdDecoder;

fn main() -> Result<()> {
    let main_model_path = "models/deepseek-coder-6.7b-base";
    let aux_model_path = "models/deepseek-coder-1.3b-base";

    println!("=== MUCD Naive 解码示例 ===\n");

    // 1. 加载 Tokenizer
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", main_model_path))
        .map_err(|e| anyhow::anyhow!("无法加载 tokenizer: {}", e))?;
    println!(
        "Tokenizer 加载完成。Vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // 2. 加载主模型
    // KV cache 显存比例：主模型用 0.7，为辅助模型预留空间
    println!("加载主模型: {} ...", main_model_path);
    let main_model = Qwen2Model::load(0, main_model_path, 0.7)?;
    println!("主模型加载完成。");

    // 3. 加载辅助模型
    // KV cache 显存比例：辅助模型用 0.9（此时剩余显存较少，尽量利用）
    println!("加载辅助模型: {} ...", aux_model_path);
    let aux_model = Qwen2Model::load(0, aux_model_path, 0.9)?;
    println!("辅助模型加载完成。");

    // 4. 创建 MUCD 解码器
    let mut decoder = MucdDecoder::new(main_model, aux_model, 0.1);

    // 5. 准备 prompt
    let prompt = "def fibonacci(n):\n    \"\"\"Calculate the n-th fibonacci number.\"\"\"\n";

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("编码失败: {}", e))?;
    let input_ids = encoding.get_ids().to_vec();

    println!("Prompt: '{}'", prompt);
    println!("Input IDs 数量: {}\n", input_ids.len());

    // 6. MUCD Naive 生成
    println!("--- MUCD Naive 生成开始 ---");
    print!("{}", prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    decoder.generate(&input_ids, 256, &tokenizer, |text, debug_info| {
        print!("{}", text);
        let _ = std::io::stdout().flush();

        // 打印调试信息到 stderr，不干扰主输出
        eprintln!(
            "[step={}, layer={}, α={:.4}, β={:.4}, H_f={:.4}, H_m={:.4}, H_a={:.4}, JS_m={:.6}, JS_a={:.6}]",
            debug_info.step,
            debug_info.selected_layer,
            debug_info.alpha,
            debug_info.beta,
            debug_info.h_final,
            debug_info.h_mid,
            debug_info.h_aux,
            debug_info.js_mid,
            debug_info.js_aux,
        );
        true // 继续生成
    })?;

    println!("\n--- 生成完成 ---");

    Ok(())
}
