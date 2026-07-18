use super::{align_face, normalize_tensor, NormalizationMethod, Recognizer, RecognizerMetadata};
use crate::{detection::FacialAreaRegion, ImageToTensor};
use burn::tensor::Tensor;

mod deepid {
    include!(concat!(env!("OUT_DIR"), "/models/recognition/deepid.rs"));
}

/// DeepID face recognition
///
/// [Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf)
pub struct DeepID {
    model: deepid::Model,
}

impl DeepID {
    /// Create a new DeepID face recognizer
    pub fn new() -> Self {
        let model = deepid::Model::default();
        Self { model: model }
    }
}

impl RecognizerMetadata for DeepID {
    const SHAPE: (u32, u32) = (47, 55);
}

impl Recognizer for DeepID {
    /// See [`super::Recognizer`].
    ///
    /// If norm is not specified it will use [`NormalizationMethod::ZeroOne`]
    fn embed<I: ImageToTensor>(
        &self,
        input: &I,
        face: FacialAreaRegion,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<1> {
        let tensor = input.to_tensor();
        let norm = norm.unwrap_or(NormalizationMethod::ZeroOne);

        let tensor = normalize_tensor(align_face(tensor, &face, Self::SHAPE), norm);

        // DeepID expects input shape as [B, H, W, C]
        let tensor = tensor.permute([0, 2, 3, 1]);

        let output = self.model.forward(tensor).squeeze();
        output
    }
}
