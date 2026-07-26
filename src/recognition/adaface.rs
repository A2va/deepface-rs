use burn::tensor::{Int, Tensor};

use super::{align_face, normalize_tensor, NormalizationMethod, Recognizer, RecognizerMetadata};
use crate::{detection::FacialAreaRegion, ImageToTensor};

#[cfg(feature = "adaface-ir101")]
mod adaface_ir101 {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/recognition/adaface_ir101.rs"
    ));
}

#[cfg(feature = "adaface-ir18")]
mod adaface_ir18 {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/recognition/adaface_ir18.rs"
    ));
}

/// Abstraction trait over Burn-generated SCRFD ONNX models.
trait AdaFaceModel: Send + Sync {
    fn forward(&self, input: Tensor<4>) -> Tensor<2>;
}

#[cfg(feature = "adaface-ir101")]
impl AdaFaceModel for adaface_ir101::Model {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        self.forward(input)
    }
}

#[cfg(feature = "adaface-ir18")]
impl AdaFaceModel for adaface_ir18::Model {
    fn forward(&self, input: Tensor<4>) -> Tensor<2> {
        self.forward(input)
    }
}

/// Generic SCRFD face detector wrapping any SCRFD model architecture.
pub struct AdaFace<M> {
    model: M,
}

impl<M: Default> AdaFace<M> {
    /// Create a detector using default model weights.
    pub fn new() -> Self {
        Self {
            model: M::default(),
        }
    }
}

/// AdaFace IR 101 face recognition.
///
/// Model and resources: [Minchul Kim - AdaFace](https://github.com/mk-minchul/AdaFace)
///
/// # Licensing
/// - Model weights: unknow license, [see](https://github.com/mk-minchul/AdaFace/issues/174)
/// - Code: [MIT License](https://github.com/mk-minchul/AdaFace/blob/master/LICENSE)
#[cfg(feature = "adaface-ir101")]
pub type AdaFaceIR101 = AdaFace<adaface_ir101::Model>;

/// AdaFace IR 18 face recognition.
///
/// Model and resources: [Minchul Kim - AdaFace](https://github.com/mk-minchul/AdaFace)
///
/// # Licensing
/// - Model weights: unknow license, [see](https://github.com/mk-minchul/AdaFace/issues/174)
/// - Code: [MIT License](https://github.com/mk-minchul/AdaFace/blob/master/LICENSE)
#[cfg(feature = "adaface-ir18")]
pub type AdaFaceIR18 = AdaFace<adaface_ir18::Model>;

impl<M> RecognizerMetadata for AdaFace<M> {
    const SHAPE: (u32, u32) = (112, 112);
}

impl<M: AdaFaceModel> Recognizer for AdaFace<M> {
    /// See [`super::Recognizer`].
    ///
    /// If norm is not specified it will use [`NormalizationMethod::ArcFace`]
    fn embed<I: ImageToTensor>(
        &self,
        input: &I,
        face: FacialAreaRegion,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<1> {
        let tensor = input.to_tensor();

        let norm = norm.unwrap_or(NormalizationMethod::ArcFace);

        let tensor = normalize_tensor(align_face(tensor, &face, Self::SHAPE), norm);

        let indices = Tensor::<1, Int>::from_data([2, 1, 0], &tensor.device());

        // AdaFace expect channels to be in BGR format
        let tensor = tensor.select(1, indices);

        let output = self.model.forward(tensor).squeeze();
        output
    }
}
