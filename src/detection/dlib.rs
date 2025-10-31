use std::marker::PhantomData;

use super::{Detector, FacialAreaRegion, ImageToTensor};
use burn::{
    prelude::{Backend, Tensor},
    tensor::Int,
};
use dlib_sys::{
    FaceDetector, FaceDetectorCnn, FaceDetectorTrait, ImageMatrix, LandmarkPredictor,
    LandmarkPredictorTrait, Rectangle,
};

/// Dlib detector model type.
pub enum DlibDetectorModel {
    /// Based on the Convolutional Neural Network (CNN).
    Cnn,
    // Based on the Histogram of Oriented Gradients (HOG).
    Hog,
}

/// Dlib face detector using the CNN model.
///
/// # Licensing
/// - Model weights: [Creative Commons CC0](https://github.com/davisking/dlib-models)
/// - Dlib library: [Boost Software License](https://github.com/davisking/dlib/blob/master/LICENSE.txt)
pub struct Dlib<B: Backend> {
    phantom: PhantomData<B>,
    model: DlibDetectorModel,
}

impl<B: Backend> Dlib<B> {
    /// Create a new Dlib face detector.
    ///
    /// If the model type is not provided it defaults to the CNN model.
    pub fn new(model: DlibDetectorModel) -> Self {
        // let model = model.unwrap_or(DlibDetectorModel::Cnn);
        Self {
            phantom: PhantomData,
            model: model,
        }
    }
}

impl<B: Backend> Detector<B> for Dlib<B> {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = None;

    /// See [`super::Detector`]
    fn detect<I: ImageToTensor<B>>(
        &self,
        input: &I,
        _confidence_threshold: f32,
        _nms_threshold: Option<f32>,
    ) -> Vec<FacialAreaRegion> {
        let device = &Default::default();
        let tensor = input.to_tensor(device).int();

        // Dlib expects u8 tensor
        let tensor = tensor.cast(burn::tensor::DType::U8);

        // Dlib expects [H, W, C] so we need to permute from [C, H, W]
        let tensor = tensor.permute([1, 2, 0]);
        let tensor_data = tensor.clone().into_data();
        let bytes = tensor_data.as_bytes();
        let ptr = bytes.as_ptr();

        let (width, height) = (tensor.dims()[1], tensor.dims()[0]);
        let matrix = unsafe { ImageMatrix::new(width, height, ptr) };

        let detector: Result<Box<dyn FaceDetectorTrait>, String> = match self.model {
            DlibDetectorModel::Cnn => {
                FaceDetectorCnn::default().map(|d| Box::new(d) as Box<dyn FaceDetectorTrait>)
            }
            DlibDetectorModel::Hog => {
                Ok(Box::new(FaceDetector::default()) as Box<dyn FaceDetectorTrait>)
            }
        };

        let Ok(detector) = detector else {
            panic!("Error loading Face Detector");
        };

        let Ok(landmarks) = LandmarkPredictor::default() else {
            panic!("Error loading Landmark Predictor");
        };

        let dets = detector.face_locations(&matrix);

        let mut results = Vec::new();
        for i in 0..dets.len() {
            let det = dets.get(i);
            let rect = det.unwrap();

            let lms = landmarks.face_landmarks(&matrix, rect);

            // Reference for the indexes
            // https://github.com/Abdelrhman-Amr-98/Head-Pose-Estimation
            // Since it is starting at 1 on the image, we need to subtract 1

            let left_eye = lms.get(42).zip(lms.get(45)).map(|(p1, p2)| {
                let x = (p1.x() + p2.x()) / 2;
                let y = (p1.y() + p2.y()) / 2;
                (x as u32, y as u32)
            });

            let right_eye = lms.get(36).zip(lms.get(39)).map(|(p1, p2)| {
                let x = (p1.x() + p2.x()) / 2;
                let y = (p1.y() + p2.y()) / 2;
                (x as u32, y as u32)
            });

            let nose = lms.get(30);
            let right_mouth = lms.get(48);
            let left_mouth = lms.get(54);

            let facial_area = FacialAreaRegion {
                x: rect.left as u32,
                y: rect.top as u32,
                w: rect.width() as u32,
                h: rect.height() as u32,
                left_eye: left_eye,
                right_eye: right_eye,
                nose: nose.map(|x| (x[0] as u32, x[1] as u32)),
                mouth_left: left_mouth.map(|x| (x[0] as u32, x[1] as u32)),
                mouth_right: right_mouth.map(|x| (x[0] as u32, x[1] as u32)),
                confidence: None,
            };
            results.push(facial_area);
        }
        results
    }
}
