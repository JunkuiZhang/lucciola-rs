use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

fn main() -> Result<()> {
    let dev = CudaContext::new(0)?;
    let stream = dev.default_stream();
    println!("GPU initialized.");

    // 1. 编译 PTX
    // 注意：这里复用了同目录下的 matmul.cu
    let ptx = compile_ptx(include_str!("./matmul.cu"))?;
    let module = dev.load_module(ptx)?;

    let kernel_simple = module.load_function("matmul")?;
    let kernel_tiled = module.load_function("matmul_tiled")?;

    // 2. 准备数据 (4096 x 4096)
    // TILE 必须与 matmul.cu 中的 #define TILE 16 保持一致
    const N: usize = 4096;
    const TILE: u32 = 16;

    println!("Preparing data ({} x {})...", N, N);
    let a_host = vec![1.0f32; N * N];
    let b_host = vec![1.0f32; N * N];

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(N * N)?; // 结果矩阵

    // 3. 配置 Launch Parameters
    // 因为 Tiled 实现依赖 Shared Memory 大小与 Block 大小一致，这里必须用 16x16
    let block = (TILE, TILE, 1);
    let grid = (
        (N as u32 + TILE - 1) / TILE,
        (N as u32 + TILE - 1) / TILE,
        1,
    );

    let cfg = LaunchConfig {
        block_dim: block,
        grid_dim: grid,
        shared_mem_bytes: 0,
    };

    println!("Launch config: Block{:?}, Grid{:?}\n", block, grid);

    // 4. 定义测试帮助函数
    let mut run_bench = |name: &str, kernel: &cudarc::driver::CudaFunction| -> Result<()> {
        // Warmup (预热，消除首次启动开销)
        for _ in 0..5 {
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&a_dev);
            builder.arg(&b_dev);
            builder.arg(&mut c_dev);
            builder.arg(&(N as i32));
            unsafe { builder.launch(cfg) }?;
        }
        stream.synchronize()?; // 等待预热完成

        // Measurement (正式计时)
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&a_dev);
            builder.arg(&b_dev);
            builder.arg(&mut c_dev);
            builder.arg(&(N as i32));
            unsafe { builder.launch(cfg) }?;
        }
        stream.synchronize()?; // 等待所有 kernel 执行完毕

        let total = start.elapsed();
        let avg = total / iters;

        println!(
            "Kernel: {:<15} | Total: {:<10.2?} | Avg: {:.2?}",
            name, total, avg
        );
        Ok(())
    };

    // 5. 运行对比
    run_bench("matmul (naive)", &kernel_simple)?;
    run_bench("matmul_tiled", &kernel_tiled)?;

    Ok(())
}
