use anyhow::Result;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

fn main() -> Result<()> {
    // 1. 初始化 GPU 设备
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // 2. 准备数据
    let n = 1024;
    let a_host = vec![1.0f32; n];
    let b_host = vec![2.0f32; n];

    // 3. 将数据从 CPU 搬运到 GPU (HtoD)
    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(n)?;

    // 4. 加载并编译内核 (JIT 编译)
    // 实际生产中我们会预编译成 PTX，入门阶段可以直接用动态字符串加载
    let ptx = compile_ptx(include_str!("./kernels/vector_add.cu"))?;
    let module = ctx.load_module(ptx)?;
    let f = module.load_function("vector_add")?;

    // 5. 配置并启动 Kernel
    let mut builder = stream.launch_builder(&f);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    builder.arg(&n);
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { builder.launch(cfg) }?;

    // 6. 将结果搬回 CPU (DtoH)
    let c_host = stream.clone_dtoh(&c_dev)?;

    println!("Result: {} ... {}", c_host[0], c_host[n - 1]); // 应该是 3.0
    Ok(())
}
