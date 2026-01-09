import os

from huggingface_hub import snapshot_download

if os.getenv('HF_ENDPOINT'):
    print(f"Using HF_ENDPOINT: {os.getenv('HF_ENDPOINT')}")

snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='./models/Qwen2.5-0.5B-Instruct',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'],
)
