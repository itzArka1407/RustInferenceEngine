use anyhow::Result;
use candle_core::{Device, Tensor};

mod tests;

fn main() -> Result<()> {
    // FIXME: For now, using CPU because the host machine doesn't have cuda-toolkit
    let a = Tensor::new(&[[1., 2.], [3., 4.], [4., 6.]], &Device::Cpu)?;
    let b = Tensor::new(&[[5., 6., 7.], [7., 8., 1.]], &Device::Cpu)?;
    let c = Tensor::new(100., &Device::Cpu)?;

    let g: Vec<Vec<f64>> = a.to_vec2()?;
    println!("{}", c.to_scalar::<f64>()?);
    println!("{g:#?}");
    println!("{:?}", a.shape()); // Shape: [length, dimension depth]

    let s = a.reshape((6, 1))?;
    println!("s: {s:?}");

    let c = a.matmul(&b)?;
    println!("{c:?}");
    Ok(())
}
