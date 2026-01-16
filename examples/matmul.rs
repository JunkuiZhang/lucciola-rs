use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(include_str!("./matmul.cu")).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("Built in {:?}", start.elapsed());

    let module = ctx.load_module(ptx)?;
    // let f = module.load_function("matmul")?;
    let f = module.load_function("matmul_tiled")?;
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.clone_htod(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let mut builder = stream.launch_builder(&f);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    builder.arg(&2i32);
    let cfg = LaunchConfig {
        block_dim: (2, 2, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }?;

    stream.memcpy_dtoh(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());

    Ok(())
}
