use cudarc::nvrtc::CompileOptions;

fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed={}/src/kernels", project_dir);
    println!(
        "cargo:rerun-if-changed={}/src/kernels/activation.cu",
        project_dir
    );
    println!(
        "cargo:rerun-if-changed={}/src/kernels/kv_cache.cu",
        project_dir
    );
    println!(
        "cargo:rerun-if-changed={}/src/kernels/rmsnorm.cu",
        project_dir
    );
    println!("cargo:rerun-if-changed={}/src/kernels/rope.cu", project_dir);
    println!(
        "cargo:rerun-if-changed={}/src/kernels/softmax.cu",
        project_dir
    );

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dst_path = std::path::Path::new(&out_dir);
    let kernels = [
        "activation.cu",
        "attention.cu",
        "kv_cache.cu",
        "rmsnorm.cu",
        "rope.cu",
        "softmax.cu",
    ];
    for kernel in kernels {
        let ptx_path = dst_path.join(format!("{}.ptx", kernel));
        if ptx_path.exists() {
            std::fs::remove_file(&ptx_path).unwrap();
        }
        println!("cargo:warning=Compiled kernel to {:?}", ptx_path);
        let source = std::fs::read_to_string(format!("src/kernels/{}", kernel)).unwrap();
        let opts = CompileOptions {
            include_paths: vec!["/usr/local/cuda/include".to_string()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&source, opts).unwrap();
        std::fs::write(ptx_path, ptx.to_src()).unwrap();
    }
}
