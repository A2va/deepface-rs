use super::{normalize_tensor, resize, NormalizationMethod, Recognizer};
use crate::ImageToTensor;
use burn::{prelude::Backend, tensor::Tensor};

mod facenet512 {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/recognition/facenet512.rs"
    ));
}

/// FaceNet512 face recognition model.
/// Model and resources: [David Sandberg - Facenet](https://github.com/davidsandberg/facenet)
///
/// Licensed under the [MIT License](https://opensource.org/licenses/MIT).  
pub struct FaceNet512<B: Backend> {
    model: facenet512::Model<B>,
}

impl<B: Backend<FloatElem = f32>> FaceNet512<B> {
    pub fn new() -> Self {
        let model = facenet512::Model::default();
        Self { model: model }
    }
}

impl<B: Backend<FloatElem = f32>> Recognizer<B> for FaceNet512<B> {
    const SHAPE: (u32, u32) = (160, 160);

    /// See [`super::Recognizer`]
    /// If norm is not specified it will use [`NormalizationMethod::FaceNet`]
    fn embed<I: ImageToTensor<B>>(
        &self,
        input: &I,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<B, 1> {
        let device = &B::Device::default();
        let tensor = input.to_tensor(device);
        let norm = norm.unwrap_or(NormalizationMethod::FaceNet);

        let tensor = normalize_tensor(resize(tensor, Self::SHAPE), norm);

        // Facenet expects input shape as  [N, H, W, C]
        let tensor = tensor.permute([0, 2, 3, 1]);
        let output = self.model.forward(tensor).squeeze(0);
        output
    }
}
