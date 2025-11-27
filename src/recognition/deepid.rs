use super::{normalize_tensor, resize, NormalizationMethod, Recognizer};
use crate::ImageToTensor;
use burn::{prelude::Backend, tensor::Tensor};

mod deepid {
    include!(concat!(env!("OUT_DIR"), "/models/recognition/deepid.rs"));
}

/// DeepID face recognition
///
/// [Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf)
pub struct DeepID<B: Backend> {
    model: deepid::Model<B>,
}

impl<B: Backend<FloatElem = f32>> DeepID<B> {
    /// Create a new DeepID face recognizer
    pub fn new() -> Self {
        let model = deepid::Model::default();
        Self { model: model }
    }
}

impl<B: Backend<FloatElem = f32>> Recognizer<B> for DeepID<B> {
    const SHAPE: (u32, u32) = (47, 55);

    /// See [`super::Recognizer`]
    /// If norm is not specified it will use [`NormalizationMethod::ZeroOne`]
    fn embed<I: ImageToTensor<B>>(
        &self,
        input: &I,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<B, 1> {
        let tensor = input.to_tensor();
        let norm = norm.unwrap_or(NormalizationMethod::ZeroOne);

        let tensor = normalize_tensor(resize(tensor, Self::SHAPE), norm);

        // DeepID expects input shape as [B, H, W, C]
        let tensor = tensor.permute([0, 2, 3, 1]);

        let output = self.model.forward(tensor).squeeze();
        output
    }
}
