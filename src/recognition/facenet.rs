use super::{align_face, normalize_tensor, NormalizationMethod, Recognizer, RecognizerMetadata};
use crate::{
    detection::FacialAreaRegion,
    ImageToTensor,
};
use burn::tensor::Tensor;

mod facenet512 {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/recognition/facenet512.rs"
    ));
}

/// FaceNet512 face recognition model.
/// Model and resources: [David Sandberg - Facenet](https://github.com/davidsandberg/facenet)
///
/// Licensed under the [MIT License](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md).
pub struct FaceNet512 {
    model: facenet512::Model,
}

impl FaceNet512 {
    pub fn new() -> Self {
        let model = facenet512::Model::default();
        Self { model: model }
    }
}

impl RecognizerMetadata for FaceNet512 {
    const SHAPE: (u32, u32) = (160, 160);
}

impl Recognizer for FaceNet512 {
    /// See [`super::Recognizer`].
    ///
    /// If norm is not specified it will use [`NormalizationMethod::FaceNet`]
    fn embed<I: ImageToTensor>(
        &self,
        input: &I,
        face: FacialAreaRegion,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<1> {
        let tensor = input.to_tensor();

        let tensor = align_face(tensor, &face, Self::SHAPE);

        let norm = norm.unwrap_or(NormalizationMethod::FaceNet);
        let tensor = normalize_tensor(tensor, norm);

        // Facenet expects input shape as  [N, H, W, C]
        let tensor = tensor.permute([0, 2, 3, 1]);
        let output = self.model.forward(tensor).squeeze();
        output
    }
}
