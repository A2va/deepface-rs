//! Dlib 68-point landmark model.

use super::{Landmarker, Landmarks, ModelKind};
use crate::{DlibDetectorModel, ImageToTensor};

/// Dlib 68-point → canonical index. Points are ordered as in dlib's
/// `full_object_detection` (jaw, right brow, left brow, nose, right eye,
/// left eye, outer mouth, inner mouth).
#[rustfmt::skip]
pub(crate) const DLIB_TO_CANONICAL: [usize; 68] = [
    // Jawline
    162, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 389,
    // Right eyebrow (image left)
    70, 63, 105, 66, 107,
    // Left eyebrow (image right)
    336, 296, 334, 293, 300,
    // Nose bridge & tip
    168, 197, 5, 4, 98, 97, 2, 326, 327,
    // Right eye (image left)
    33, 160, 158, 133, 153, 144,
    // Left eye (image right)
    362, 385, 387, 263, 373, 380,
    // Outer mouth
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    // Inner mouth
    78, 82, 13, 312, 308, 317, 14, 87,
];

/// A face bounding box produced by dlib's face detector.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub confidence: f32,
}

/// Dlib 68-point landmark model (self-contained detection + predictor).
///
/// # Licensing
/// - Model weights: [Creative Commons CC0](https://github.com/davisking/dlib-models)
/// - Dlib library: [Boost Software License](https://github.com/davisking/dlib/blob/master/LICENSE.txt)
pub struct DlibLandmarks {
    detection: Box<dyn dlib_sys::FaceDetectorTrait>,
    predictor: dlib_sys::LandmarkPredictor,
}

impl DlibLandmarks {
    /// Create a new dlib landmark model with a given detector type.
    pub fn new(model: DlibDetectorModel) -> Self {
        use dlib_sys::{FaceDetector, FaceDetectorCnn, FaceDetectorTrait, LandmarkPredictor};

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
        let Ok(predictor) = LandmarkPredictor::default() else {
            panic!("Error loading Landmark Predictor");
        };
        Self { detection, predictor }
    }

    /// Detect faces, returning each face's landmarks and its detector bbox.
    pub(crate) fn detect_faces<I: ImageToTensor>(&self, input: &I) -> Vec<(Landmarks, FaceRect)> {
        use dlib_sys::{ImageMatrix, LandmarkPredictorTrait};

        let tensor = input.to_tensor().int();
        let tensor = tensor.cast(burn::tensor::DType::U8);
        let tensor = tensor.permute([1, 2, 0]);
        let tensor_data = tensor.clone().into_data();
        let bytes = tensor_data.as_bytes();
        let ptr = bytes.as_ptr();
        let (width, height) = (tensor.dims()[1], tensor.dims()[0]);
        let matrix = unsafe { ImageMatrix::new(width, height, ptr) };

        let dets = self.detection.face_locations(&matrix);
        dets.iter()
            .map(|rect| {
                let lms = self.predictor.face_landmarks(&matrix, rect);
                let points: Vec<(f32, f32)> = lms
                    .into_iter()
                    .map(|m| (m.x() as f32, m.y() as f32))
                    .collect();
                let landmarks = Landmarks::new(points, ModelKind::Dlib68);
                let face_rect = FaceRect {
                    x: rect.left as u32,
                    y: rect.top as u32,
                    w: rect.width() as u32,
                    h: rect.height() as u32,
                    confidence: rect.confidence as f32,
                };
                (landmarks, face_rect)
            })
            .collect()
    }
}

impl Landmarker for DlibLandmarks {
    fn predict<I: ImageToTensor>(&self, input: &I) -> Vec<Landmarks> {
        self.detect_faces(input)
            .into_iter()
            .map(|(landmarks, _)| landmarks)
            .collect()
    }
}