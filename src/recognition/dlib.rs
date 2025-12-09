use std::marker::PhantomData;

use burn::{prelude::Backend, Tensor};
use dlib_sys::{
    FaceDetector, FaceDetectorCnn, FaceDetectorTrait, FaceEncoderNetwork, FaceEncoderTrait,
    ImageMatrix, LandmarkPredictor, LandmarkPredictorTrait,
};

use super::{NormalizationMethod, Recognizer};
use crate::{DlibDetectorModel, ImageToTensor};

/// Dlib face recognition using CNN model.
///
/// # Licensing
/// - Model weights: [Creative Commons CC0](https://github.com/davisking/dlib-models)
/// - Dlib library: [Boost Software License](https://github.com/davisking/dlib/blob/master/LICENSE.txt)
pub struct DlibRecognition<B: Backend> {
    phantom: PhantomData<B>,
    detection: Box<dyn FaceDetectorTrait>,
    landmarks: LandmarkPredictor,
    recognition: FaceEncoderNetwork,
}

impl<B: Backend> DlibRecognition<B> {
    /// Create a new Dlib face recognition with a given model type.
    ///
    /// Since this model requires the landmarks provided by the dlib detection model,
    /// this embedding model can support images containing more than just a close-up of a face.
    ///
    /// Burn backend are not supported on this model.
    pub fn new(model: DlibDetectorModel) -> Self {
        let Ok(recognition) = FaceEncoderNetwork::default() else {
            panic!("Error loading Face Encoder.");
        };

        let detector: Result<Box<dyn FaceDetectorTrait>, String> = match model {
            DlibDetectorModel::Cnn => {
                FaceDetectorCnn::default().map(|d| Box::new(d) as Box<dyn FaceDetectorTrait>)
            }
            DlibDetectorModel::Hog => {
                Ok(Box::new(FaceDetector::default()) as Box<dyn FaceDetectorTrait>)
            }
        };

        let Ok(detection) = detector else {
            panic!("Error loading Face Detector.");
        };

        let Ok(landmarks) = LandmarkPredictor::default() else {
            panic!("Error loading Landmark Predictor");
        };

        Self {
            phantom: PhantomData,
            recognition: recognition,
            detection: detection,
            landmarks: landmarks,
        }
    }
}

impl<B: Backend> Recognizer<B> for DlibRecognition<B> {
    const SHAPE: (u32, u32) = (0, 0);

    /// See [`super::Recognizer`].
    ///
    /// In this model the norm parameter is not used.
    ///
    /// If the input image has already been cropped to include the face,
    /// this can trigger a panic because the integrated detection model hasn't found a face.
    /// In this case, try providing the full image directly.
    fn embed<I: ImageToTensor<B>>(
        &self,
        input: &I,
        _norm: Option<NormalizationMethod>,
    ) -> Tensor<B, 1> {
        let tensor = input.to_tensor().int();

        // Dlib expects u8 tensor
        let tensor = tensor.cast(burn::tensor::DType::U8);

        // Dlib expects [H, W, C] so we need to permute from [C, H, W]
        let tensor = tensor.permute([1, 2, 0]);
        let tensor_data = tensor.clone().into_data();
        let bytes = tensor_data.as_bytes();
        let ptr = bytes.as_ptr();

        let (width, height) = (tensor.dims()[1], tensor.dims()[0]);
        let matrix = unsafe { ImageMatrix::new(width, height, ptr) };

        let dets = self.detection.face_locations(&matrix);
        let rect = dets.first().unwrap();
        let landmarks = self.landmarks.face_landmarks(&matrix, rect);

        let encodings = self
            .recognition
            .get_face_encodings(&matrix, &[landmarks], 0);
        let embeddings = encodings.first().unwrap();

        let device = &Default::default();
        Tensor::<B, 1>::from_floats(embeddings.as_ref(), &device)
    }
}
