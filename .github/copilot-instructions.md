# Lucciola Codebase Instructions

Lucciola is a high-performance LLM inference engine written in Rust using CUDA via `cudarc`. It directly manages GPU memory and executes custom CUDA kernels.

## Core Architecture

- **Language**: Rust (Interface/Logic) + CUDA C++ (Kernels).
- **GPU Interaction**: Uses `cudarc` for managing context, streams, and launching kernels.
- **Model Storage**: Loads `safetensors` files using memory mapping (`memmap2`) for zero-copy weight loading.
- **Kernels**:
  - Source: `src/kernels/*.cu`
  - Compilation: Managed by `build.rs` using `nvrtc` to generate PTX files in `OUT_DIR`.
  - Linking: `src/ptx.rs` embeds PTX content via `include_str!`.
  - Loading: `src/kernels.rs` loads specific function symbols (e.g., `silu_and_mul_kernel`) into a `CudaFunctions` struct.

## Developer Workflows

### Setup
1. **Model Download**:
   Use the provided script to fetch model weights (e.g., Qwen2.5-0.5B):
   ```bash
   pip install -U "huggingface_hub[cli]"
   python scripts/download_qwen.py
   ```

2. **Build**:
   Standard `cargo build` works, but requires CUDA toolkit installed. `build.rs` handles kernel compilation.

### Running Examples
The primary entry points for testing and benchmarking are in `examples/`:
- **Inference Benchmark**: `cargo run --release --example bench_inference`
- **Matrix Multiplication**: `cargo run --release --example bench_matmul`

## Code Conventions

- **Precision**: Uses `half::bf16` (Brain Floating Point) extensively for model weights and computation.
- **Error Handling**: Uses `anyhow::Result` for application-level error management.
- **Buffers**: Manages GPU memory via `cudarc::driver::CudaSlice<T>`.
- **Kernel Changes**:
  - IF you edit `.cu` files, `build.rs` detects changes and recompiles.
  - IF you add a NEW kernel:
    1. Create `src/kernels/new_kernel.cu`.
    2. Add to `kernels` array in `build.rs`.
    3. Add PTX loader in `src/ptx.rs`.
    4. Add loader logic in `src/kernels.rs`.

## Critical Files
- `src/models.rs`: Defines model architecture (`ModelConfig`, `LayerWeights`) and orchestrates layer execution.
- `src/kernels.rs`: Registry for loaded CUDA functions.
- `build.rs`: Compiles CUDA code to PTX. Ensure this is updated when adding new kernels.
