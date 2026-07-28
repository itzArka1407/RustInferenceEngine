use candle_core::{Device, Tensor};

#[test]
fn test_tensors() -> anyhow::Result<()> {
    let a = Tensor::new(&[[1., 2.], [3., 4.], [4., 6.]], &Device::Cpu)?;
    let b = Tensor::new(&[[5., 6., 7.], [7., 8., 1.]], &Device::Cpu)?;
    let c = Tensor::new(100., &Device::Cpu)?;

    let g: Vec<Vec<f64>> = a.to_vec2()?;
    println!("{}", c.to_scalar::<f64>()?);
    println!("{g:?}");
    println!("{:?}", a.shape()); // Shape: [length, dimension depth]

    let s = a.reshape((6, 1))?;
    println!("s: {s:?}");

    let c = a.matmul(&b)?;
    let d = c.transpose(0, 1)?;
    let e = c.unsqueeze(0)?;
    let f = e.squeeze(0)?; // c.squeeze(0) works same bcoz axis 0 for c isn't of size: 1 -- squeeze
    // only works if size of target axis = 1
    let g = c.narrow(1, 1, 2)?;
    println!(
        "c: {0:?}\nd: {1:?}\ne: {2:?}\nf: {3:?}\ng: {4:?}",
        c.to_vec2::<f64>()?,
        d.to_vec2::<f64>()?,
        e.to_vec3::<f64>()?,
        f.to_vec2::<f64>()?,
        g.to_vec2::<f64>()?
    );

    Ok(())
}

// Simplied linear operations on models -- Y = XW + B
// Output = input * weight + bias
#[test]
fn linear_operations() -> anyhow::Result<()> {
    let x = Tensor::new(&[[1., 3.], [2., 5.]], &Device::Cpu)?; // Replicate a [2,2] input
    let w = Tensor::new(&[[2., 3., 1.], [5., 1., 9.]], &Device::Cpu)?; // Replicate a [2, 3] weight
    let b = Tensor::new(&[4., 11., 16.3], &Device::Cpu)?; // The bias of the operation

    let final_output = x.matmul(&w)?.broadcast_add(&b)?;
    println!("X: {:?} | Sh: {:?}", x.to_vec2::<f64>()?, x.shape());
    println!("W: {:?} | Sh: {:?}", w.to_vec2::<f64>()?, w.shape());
    println!("B: {:?} | Sh: {:?}", b.to_vec1::<f64>()?, b.shape());
    println!(
        "final: {:?} | Sh: {:?}",
        final_output.to_vec2::<f64>()?,
        final_output.shape()
    );
    Ok(())
}
