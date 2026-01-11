use anyhow::Result;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug)]
pub enum ChatFormat {
    ChatML, // Qwen, etc (<|im_start|>...)
    Llama2, // [INST] ... [/INST]
    Raw,    // No formatting
}

impl ChatFormat {
    pub fn from_model_type(model_type: &str) -> Self {
        match model_type {
            "qwen2" | "qwen" => ChatFormat::ChatML,
            "llama" => ChatFormat::Llama2, // A crude approximation
            _ => ChatFormat::Raw,
        }
    }
}

pub struct ChatTemplate {
    messages: Vec<Message>,
    format: ChatFormat,
}

impl ChatTemplate {
    pub fn new(model_type: Option<&str>) -> Self {
        let format = match model_type {
            Some(t) => ChatFormat::from_model_type(t),
            None => ChatFormat::Raw,
        };

        Self {
            messages: Vec::new(),
            format,
        }
    }

    pub fn add(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn apply(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        match self.format {
            ChatFormat::ChatML => self.apply_chatml(tokenizer),
            ChatFormat::Llama2 => self.apply_llama2(tokenizer),
            ChatFormat::Raw => self.apply_raw(tokenizer),
        }
    }

    fn apply_chatml(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        // Qwen2.5 ChatML format
        // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n

        let mut ids = Vec::new();

        // Why manual IDs?
        // Standard `tokenizer.encode` treats input as raw text. If we passed "<|im_start|>" as string,
        // it might be split into ["<", "|", "im", ...] instead of the single control token.
        // Direct ID injection ensures the model sees the correct special control tokens.
        let im_start = tokenizer.token_to_id("<|im_start|>").unwrap_or(151644);
        let im_end = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        let newline = tokenizer.token_to_id("\n").unwrap_or(198);

        for msg in &self.messages {
            ids.push(im_start);

            // Encode role + newline
            let role_bytes = format!("{}\n", msg.role);
            ids.extend(
                tokenizer
                    .encode(role_bytes, false)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .get_ids(),
            );

            // Encode content
            ids.extend(
                tokenizer
                    .encode(msg.content.as_str(), false)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .get_ids(),
            );

            ids.push(im_end);
            ids.push(newline);
        }

        // Prepare for assistant generation
        if let Some(last) = self.messages.last() {
            if last.role != "assistant" {
                ids.push(im_start);
                let role_bytes = "assistant\n";
                ids.extend(
                    tokenizer
                        .encode(role_bytes, false)
                        .map_err(|e| anyhow::anyhow!(e))?
                        .get_ids(),
                );
            }
        }

        Ok(ids)
    }

    fn apply_llama2(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        // Placeholder for Llama-2/3 format
        // [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}
        // This is a simplified version, usually needs BOS/EOS handling
        let mut ids = Vec::new();
        for msg in &self.messages {
            ids.extend(
                tokenizer
                    .encode(format!("{}: {}\n", msg.role, msg.content), false)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .get_ids(),
            );
        }
        Ok(ids)
    }

    fn apply_raw(&self, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for msg in &self.messages {
            ids.extend(
                tokenizer
                    .encode(msg.content.as_str(), false)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .get_ids(),
            );
        }
        Ok(ids)
    }
}
