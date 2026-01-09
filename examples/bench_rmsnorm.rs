use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::time::Instant;

fn main() -> Result<()> {
    let dev = CudaContext::new(0)?;
    let stream = dev.default_stream();
    println!("GPU initialized.");

    // 1. 编译 PTX
    let opts = CompileOptions {
        include_paths: vec!["/usr/local/cuda/include".to_string()],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(include_str!("./rmsnorm.cu"), opts)?;
    let module = dev.load_module(ptx)?;

    let kernel_naive = module.load_function("rmsnorm_kernel")?;
    let kernel_opt = module.load_function("rmsnorm_optimized")?;
    let kernel_nvidia = module.load_function("rmsnorm_nvidia")?;

    // 2. Setup Data
    // 模拟 Llama-7B 规模: 4096 hidden size
    const HIDDEN_SIZE: usize = 4096;
    // 批处理大小 (Tokens)
    const BATCH_SIZE: usize = 4096;

    println!(
        "Benchmarking RMSNorm with [Batch={}, Hidden={}]",
        BATCH_SIZE, HIDDEN_SIZE
    );

    let input_host = vec![1.0f32; BATCH_SIZE * HIDDEN_SIZE];
    let weight_host = vec![0.5f32; HIDDEN_SIZE];

    let input_dev = stream.clone_htod(&input_host)?;
    let weight_dev = stream.clone_htod(&weight_host)?;
    let mut out_dev = stream.alloc_zeros::<f32>(BATCH_SIZE * HIDDEN_SIZE)?;

    // 3. Launch Config
    // RMSNorm 每个 Block 处理一行 (Row / Token)
    // 使用 1024 线程以最大化利用 Block 内资源
    let block_size = 1024;
    let cfg = LaunchConfig {
        grid_dim: (BATCH_SIZE as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let epsilon = 1e-6f32;
    let n_cols = HIDDEN_SIZE as i32;

    // 4. Bench Function
    let mut run_bench = |name: &str, kernel: &cudarc::driver::CudaFunction| -> Result<()> {
        // Warmup
        for _ in 0..10 {
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&mut out_dev);
            builder.arg(&input_dev);
            builder.arg(&weight_dev);
            builder.arg(&epsilon);
            builder.arg(&n_cols);
            unsafe { builder.launch(cfg) }?;
        }
        stream.synchronize()?;

        let iters = 200;
        let start = Instant::now();
        for _ in 0..iters {
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&mut out_dev);
            builder.arg(&input_dev);
            builder.arg(&weight_dev);
            builder.arg(&epsilon);
            builder.arg(&n_cols);
            unsafe { builder.launch(cfg) }?;
        }
        stream.synchronize()?;

        let total = start.elapsed();
        let avg = total / iters;

        // 估算有效带宽 (Effective Bandwidth)
        // 理论上 RMSNorm 必须读取一次 Input (4 bytes), 进行归约后再读取一次 Input (或缓存), 写入一次 Output (4 bytes)
        // 最理想情况按 Read+Write 算: 2 * 4 bytes * Elements
        let bytes = (BATCH_SIZE * HIDDEN_SIZE * 2 * 4) as f64;
        let gb_per_sec = (bytes / 1e9) / avg.as_secs_f64();

        println!(
            "Kernel: {:<20} | Avg: {:<10.2?} | Bandwidth: {:.2} GB/s",
            name, avg, gb_per_sec
        );
        Ok(())
    };

    run_bench("rmsnorm_kernel", &kernel_naive)?;
    run_bench("rmsnorm_optimized", &kernel_opt)?;
    run_bench("rmsnorm_nvidia", &kernel_nvidia)?;

    Ok(())
}
