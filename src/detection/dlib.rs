use burn::vision::NmsOptions;

use super::{Detector, DetectorMetadata, FacialAreaRegion};
use crate::landmarks::dlib::DlibLandmarks;
use crate::{DlibDetectorModel, ImageToTensor};

/// Dlib face detector, reusing the [`DlibLandmarks`] detection pipeline.
///
/// The bounding box comes from dlib's face detector; the 68-point landmarks
/// are exposed alongside it.
///
/// # Licensing
/// - Model weights: [Creative Commons CC0](https://github.com/davisking/dlib-models)
/// - Dlib library: [Boost Software License](https://github.com/davisking/dlib/blob/master/LICENSE.txt)
pub struct DlibDetection {
    landmarks: DlibLandmarks,
}

impl DlibDetection {
    /// Create a new Dlib face detector with a given model type.
    ///
    /// Burn backend are not supported on this model.
    pub fn new(model: DlibDetectorModel) -> Self {
        Self {
            landmarks: DlibLandmarks::new(model),
        }
    }
}

impl DetectorMetadata for DlibDetection {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = None;
}

impl Detector for DlibDetection {
    /// See [`super::Detector`].
    ///
    /// Unlike most models, which return a confidence score between 0 and 1,
    /// Dlib can return a value lower than 0, up to a maximum of 3.5, based on this [issue](https://github.com/davisking/dlib/issues/761).
    /// Also the nms options are not supported for this model.
    fn detect<I: ImageToTensor>(
        &self,
        input: &I,
        _nms_options: NmsOptions,
    ) -> Vec<FacialAreaRegion> {
        self.landmarks
            .detect_faces(input)
            .into_iter()
            .map(|(landmarks, rect)| FacialAreaRegion {
                x: rect.x,
                y: rect.y,
                w: rect.w,
                h: rect.h,
                confidence: Some(rect.confidence),
                landmarks: Some(landmarks),
            })
            .collect()
    }
}