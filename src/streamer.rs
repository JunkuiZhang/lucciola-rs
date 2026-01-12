use tokenizers::Tokenizer;

pub struct Streamer<'a> {
    tokenizer: &'a Tokenizer,
    pending: Vec<u32>,
}

impl<'a> Streamer<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::new(),
        }
    }

    pub fn put(&mut self, token: u32) -> Option<String> {
        self.pending.push(token);

        // Decoding with skip_special_tokens=true
        if let Ok(text) = self.tokenizer.decode(&self.pending, true) {
            // If the text ends with a replacement character, it's likely an incomplete UTF-8 sequence.
            // We keep accumulating tokens.
            // However, to prevent getting stuck if the model actually generates a replacement char,
            // or if the buffer grows too large (UTF-8 max is 4 bytes, so > 5 tokens is suspicious),
            // we force flush.
            if self.pending.len() <= 6 && text.ends_with(char::REPLACEMENT_CHARACTER) {
                return None;
            }

            // Valid sequence or forced flush
            self.pending.clear();
            return Some(text);
        }

        // Decoding failed (unlikely), keep pending
        None
    }
}
