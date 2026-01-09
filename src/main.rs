use anyhow::Result;
use cudarc::driver::CudaContext;

use crate::models::Qwen2Model;

mod models;

fn main() -> Result<()> {
    let device = CudaContext::new(0)?;
    let path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct";
    let model = Qwen2Model::load(&device, path)?;
    println!("Model loaded successfully.");
    std::thread::sleep(std::time::Duration::from_secs(5));
    Ok(())
}
