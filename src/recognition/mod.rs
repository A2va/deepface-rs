#[cfg(feature = "facenet512")]
pub mod facenet;
#[cfg(feature = "facenet512")]
pub use crate::recognition::facenet::FaceNet512;

#[cfg(feature = "deepid")]
pub mod deepid;
#[cfg(feature = "deepid")]
pub use crate::recognition::deepid::DeepID;

pub mod verify;
pub use verify::*;

use crate::ImageToTensor;
use burn::{
    nn::interpolate::Interpolate2dConfig,
    prelude::{Backend, Tensor, ToElement},
    tensor::s,
};

#[derive(Clone, Copy, Debug)]
pub enum RecognitionModel {
    #[cfg(feature = "deepid")]
    DeepID,
    #[cfg(feature = "facenet512")]
    FaceNet512,
}

/// A trait that all face recognition models implements
pub trait Recognizer<B: Backend> {
    /// The expected input shape of the model.
    const SHAPE: (u32, u32);

    /// Generate an embedding from an input image, applying the specified normalization method if provided.
    fn embed<I: ImageToTensor<B>>(
        &self,
        input: &I,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<B, 1>;
}

/// Normalization methods for face embeddings
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Normalize to [0, 1]
    ZeroOne,
    /// Normalize for the FaceNet model
    FaceNet,
    /// Normalize for the FaceNet2018 model
    FaceNet2018,
    /// Normalize for the VGGFace model
    VGGFace,
    /// Normalize for the VGGFace2 model
    VGGFace2,
    /// Normalize for the ArcFace model
    ArcFace,
}

fn resize<B: Backend>(
    tensor: Tensor<B, 3>, // [C, H, W]
    shape: (u32, u32),
) -> Tensor<B, 4> {
    let (target_h, target_w) = (shape.0 as usize, shape.1 as usize);

    let interpolate = Interpolate2dConfig::new()
        .with_output_size(Some([target_h, target_w]))
        .init();
    let resized = interpolate.forward(tensor.unsqueeze::<4>()); // [1, C, H, W]

    assert!(resized.dims() == [1, 3, target_h, target_w]);
    resized
}

fn normalize_tensor<B: Backend>(tensor: Tensor<B, 4>, norm: NormalizationMethod) -> Tensor<B, 4> {
    // Check that the tensor is between 0 and 255, rgb image
    let max = tensor.clone().max().into_scalar().to_i32();
    let min = tensor.clone().min().into_scalar().to_i32();
    assert!(max <= 255);
    assert!(min >= 0);

    match norm {
        NormalizationMethod::None => tensor,
        NormalizationMethod::ZeroOne => tensor / 255.0,
        NormalizationMethod::FaceNet => {
            let flat = tensor.clone().flatten::<2>(1, -1); // [N, C*H*W]

            // Compute mean & std per batch → shape [N,1,1,1] for broadcasting
            let mean = flat
                .clone()
                .mean_dim(1)
                .reshape([tensor.dims()[0], 1, 1, 1]);
            let std = flat.var(1).sqrt().reshape([tensor.dims()[0], 1, 1, 1]);
            (tensor - mean) / std
        }
        NormalizationMethod::FaceNet2018 => (tensor / 127.5) - 1.0,
        NormalizationMethod::VGGFace => normalize_with_means(tensor, [93.5940, 104.7624, 129.1863]),
        NormalizationMethod::VGGFace2 => {
            normalize_with_means(tensor, [91.4953, 103.8827, 131.0912])
        }
        NormalizationMethod::ArcFace => (tensor - 127.5) / 128.0,
    }
}

// Excepct a tensor in the format [B, C, H, W]
fn normalize_with_means<B: Backend>(tensor: Tensor<B, 4>, means: [f64; 3]) -> Tensor<B, 4> {
    // Split channels and subtract
    let mut out = tensor.clone();
    for (c, mean) in means.iter().enumerate() {
        let channel = out.clone().slice(s![.., c..c + 1, .., ..]);
        let channel: Tensor<B, 4> = channel - *mean;
        out = out.slice_assign(s![.., c..c + 1, .., ..], channel);
    }

    out
}
