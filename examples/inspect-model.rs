use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let file_path = "/app/lucciola/models/Qwen2.5-0.5B-Instruct/model.safetensors";
    let file = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let tensors = SafeTensors::deserialize(&mmap)?;

    println!(
        "{:<60} | {:<20} | {:<10}",
        "Tensor Name", "Shape", "Data Type"
    );
    println!("{}", "-".repeat(95));

    for (name, view) in tensors.tensors() {
        println!("{:<60} | {:<20?} | {:?}", name, view.shape(), view.dtype());
    }

    Ok(())
}
