use cudarc::nvrtc::CompileOptions;

fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed={project_dir}/src/kernels");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/activation.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/kv_cache.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/rmsnorm.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/rope.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/embedding.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/sampling.cu");
    println!("cargo:rerun-if-changed={project_dir}/src/kernels/sampling_optimized.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dst_path = std::path::Path::new(&out_dir);
    let kernels = [
        "activation.cu",
        "attention.cu",
        "embedding.cu",
        "kv_cache.cu",
        "rmsnorm.cu",
        "rope.cu",
        "sampling.cu",
        "sampling_optimized.cu",
    ];
    for kernel in kernels {
        let ptx_path = dst_path.join(format!("{kernel}.ptx"));
        if ptx_path.exists() {
            std::fs::remove_file(&ptx_path).unwrap();
        }
        let source = std::fs::read_to_string(format!("src/kernels/{kernel}")).unwrap();
        let opts = CompileOptions {
            include_paths: vec![
                "/opt/cuda/include".to_string(),
                "/usr/include/cccl".to_string(),
                "/usr/local/cuda/include".to_string(),
            ],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&source, opts).unwrap();
        std::fs::write(ptx_path, ptx.to_src()).unwrap();
    }
}
