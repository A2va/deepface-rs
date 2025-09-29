use burn::{prelude::Backend, tensor::Tensor};
use tuple_conv::RepeatedTuple;

use super::{
    non_maximum_suppression, resize_tensor, BoundingBox, Detector, FacialAreaRegion, Landmarks,
    ResizedDimensions,
};
use crate::ImageToTensor;

mod yunet {
    include!(concat!(env!("OUT_DIR"), "/models/detection/yunet.rs"));
}

/// Yunet face detector.
///
/// A lightweight fast face detection model trained by OpenCV.  
/// It predicts face bounding boxes and landmarks from an input image.  
///
/// Model and resources: [OpenCV Zoo – Yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)  
///
/// # Licensing
/// - Model weights: [MIT License](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)  
/// - OpenCV reference implementation: [Apache 2.0 License](https://github.com/opencv/opencv/blob/4.x/LICENSE)
/// # Reference:
///
/// ```text
/// @article{wu2023yunet,
///  title={Yunet: A tiny millisecond-level face detector},
///  author={Wu, Wei and Peng, Hanyang and Yu, Shiqi},
///  journal={Machine Intelligence Research},
///  volume={20},
///  number={5},
///  pages={656--665},
///  year={2023},
///  publisher={Springer}
/// }
/// ```
pub struct Yunet<B: Backend> {
    model: yunet::Model<B>,
}

impl<B: Backend<FloatElem = f32>> Yunet<B> {
    /// Create a new Yunet face detector.
    pub fn new() -> Self {
        let model = yunet::Model::default();
        Self { model }
    }

    // Original implementation
    // https://github.com/opencv/opencv/blob/829495355d7da3f073828dd584f1cdba9e07dc65/modules/objdetect/src/face_detect.cpp
    fn postprocess(
        &self,
        outputs: Vec<Tensor<B, 3>>,
        sizes: ResizedDimensions,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<[(f32, f32); 5]>) {
        let (mut dets, mut lms) = self.decode(outputs, sizes, confidence_threshold, nms_threshold);

        dets = dets
            .into_iter()
            .map(|mut bbbox| {
                bbbox.xmin /= sizes.width_scale;
                bbbox.xmax /= sizes.width_scale;
                bbbox.ymin /= sizes.height_scale;
                bbbox.ymax /= sizes.height_scale;
                bbbox
            })
            .collect();

        // Scale landmarks
        lms = lms
            .into_iter()
            .map(|mut landmark| {
                for i in 0..5 {
                    landmark[i] = (
                        landmark[i].0 / sizes.width_scale,
                        landmark[i].1 / sizes.height_scale,
                    )
                }
                landmark
            })
            .collect();

        (dets, lms)
    }

    fn decode(
        &self,
        outputs: Vec<Tensor<B, 3>>,
        sizes: ResizedDimensions,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let strides = [8, 16, 32];

        let mut boxes = Vec::new();
        let mut lms = Vec::new();

        for (i, stride) in strides.iter().enumerate() {
            let cls = &outputs[i];
            let obj = &outputs[i + strides.len() * 1];
            let bbox = &outputs[i + strides.len() * 2];
            let kkps = &outputs[i + strides.len() * 3];

            let rows: usize = (sizes.height as usize / stride) as usize;
            let cols = (sizes.width as usize / stride) as usize;

            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;

                    let cls_score = cls.clone().slice([0, idx, 0]).into_scalar();
                    let obj_score = obj.clone().slice([0, idx, 0]).into_scalar();

                    let cls_score = cls_score.min(1.0).max(0.0);
                    let obj_score = obj_score.min(1.0).max(0.0);

                    let score = f32::sqrt(cls_score * obj_score);
                    // Check if the score meets the threshold
                    if score < confidence_threshold {
                        continue;
                    }

                    let cx = (col as f32 + bbox.clone().slice([0, idx, 0]).into_scalar())
                        * (*stride as f32);
                    let cy = (row as f32 + bbox.clone().slice([0, idx, 1]).into_scalar())
                        * (*stride as f32);
                    let w =
                        f32::exp(bbox.clone().slice([0, idx, 2]).into_scalar()) * (*stride as f32);
                    let h =
                        f32::exp(bbox.clone().slice([0, idx, 3]).into_scalar()) * (*stride as f32);

                    let x1 = cx - w / 2.0;
                    let y1 = cy - h / 2.0;

                    let x2 = cx + w / 2.0;
                    let y2 = cy + h / 2.0;
                    boxes.push(BoundingBox {
                        xmin: x1,
                        ymin: y1,
                        xmax: x2,
                        ymax: y2,
                        confidence: score,
                    });

                    let mut lm: Landmarks = [(0.0, 0.0); 5];
                    // Get landmarks
                    for n in 0..5 {
                        let landmark_x = (kkps.clone().slice([0, idx, n * 2]).into_scalar()
                            + col as f32)
                            * (*stride as f32);
                        let landmark_y = (kkps.clone().slice([0, idx, n * 2 + 1]).into_scalar()
                            + row as f32)
                            * (*stride as f32);

                        lm[n] = (landmark_x, landmark_y);
                    }
                    lms.push(lm);
                }
            }
        }

        non_maximum_suppression(&mut boxes, &mut lms, nms_threshold);
        (boxes, lms)
    }
}

impl<B: Backend<FloatElem = f32>> Detector<B> for Yunet<B> {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = Some(640);

    /// See [`super::Detector`]
    fn detect<I: ImageToTensor<B>>(
        &self,
        input: &I,
        confidence_threshold: f32,
    ) -> Vec<FacialAreaRegion> {
        let nms_threshold = 0.3;
        let device = &Default::default();
        let (tensor, sizes) = resize_tensor(input.to_tensor(device), Self::DIVISOR, Self::MAX_SIZE);

        let outputs = self.model.forward(tensor).to_vec();

        let (detections, lms) =
            self.postprocess(outputs, sizes, confidence_threshold, nms_threshold);

        let mut results = Vec::new();
        for (i, detection) in detections.iter().enumerate() {
            let x = detection.xmin;
            let y = detection.ymin;
            let w = detection.xmax - x;
            let h = detection.ymax - y;

            let landmark = &lms[i];

            let right_eye = (landmark[0].0 as u32, landmark[0].1 as u32);
            let left_eye = (landmark[1].0 as u32, landmark[1].1 as u32);

            let nose = (landmark[2].0 as u32, landmark[2].1 as u32);
            let right_mouth = (landmark[3].0 as u32, landmark[3].1 as u32);
            let left_mouth = (landmark[4].0 as u32, landmark[4].1 as u32);

            let confidence = detection.confidence.min(0.0).max(1.0);
            let facial_area = FacialAreaRegion {
                x: x as u32,
                y: y as u32,
                w: w as u32,
                h: h as u32,
                left_eye: Some(left_eye),
                right_eye: Some(right_eye),
                nose: Some(nose),
                mouth_right: Some(right_mouth),
                mouth_left: Some(left_mouth),
                confidence: Some(confidence),
            };
            results.push(facial_area);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use crate::detection::{Detector, Yunet};
    use burn::backend::NdArray;

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");
        let model: Yunet<NdArray> = Yunet::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(&img, 0.8);

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
