use std::ops::Bound;

use burn::{backend::ndarray::NdArray, tensor::Tensor};
use image::DynamicImage;
use tuple_conv::RepeatedTuple;

use super::{
    non_maximum_suppression, resize_to_multiple_of_divisor, to_tensor, BoundingBox, Landmarks,
};
use super::{DeepFaceBackend, Detector, FacialAreaRegion};

mod yunet {
    include!(concat!(env!("OUT_DIR"), "/models/detection/yunet.rs"));
}

// https://github.com/opencv/opencv/blob/829495355d7da3f073828dd584f1cdba9e07dc65/modules/objdetect/src/face_detect.cpp#L20


/// Yunet face detector.
///
/// A lightweight fast face detection model trained by OpenCV.  
/// It predicts face bounding boxes and landmarks from an input image.  
///
/// Model and resources: [OpenCV Zoo – Yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)  
///
/// # Licensing
/// - Model weights: [MIT License](https://opensource.org/licenses/MIT)  
/// - OpenCV reference implementation: [Apache 2.0 License](https://opensource.org/license/apache-2-0)
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
pub struct Yunet {
    model: yunet::Model<DeepFaceBackend>,
}

impl Yunet {
    /// Create a new Yunet face detector.
    pub fn new() -> Self {
        let model = yunet::Model::default();
        Self { model }
    }

    fn postprocess(
        &self,
        outputs: Vec<Tensor<DeepFaceBackend, 3>>,
        sizes: (u32, u32, f32, f32),
    ) -> (Vec<BoundingBox>, Vec<[(f32, f32); 5]>) {
        let (mut dets, mut lms) = self.decode(outputs, sizes);

        let (height, width, scale_h, scale_w) = sizes;

        if !dets.is_empty() {
            dets = dets
                .into_iter()
                .map(|mut bbbox| {
                    bbbox.xmin /= scale_w;
                    bbbox.xmax /= scale_w;
                    bbbox.ymin /= scale_h;
                    bbbox.ymax /= scale_h;
                    bbbox
                })
                .collect::<Vec<BoundingBox>>();

            // Scale landmarks
            lms = lms
                .into_iter()
                .map(|mut landmark| {
                    for i in 0..5 {
                        landmark[i] = (landmark[i].0 / scale_w, landmark[i].1 / scale_h)
                    }
                    landmark
                })
                .collect::<Vec<Landmarks>>();
        }
        (dets, lms)
    }

    fn decode(
        &self,
        outputs: Vec<Tensor<DeepFaceBackend, 3>>,
        sizes: (u32, u32, f32, f32),
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let strides = [8, 16, 32];

        let mut boxes = Vec::new();
        let mut lms = Vec::new();

        let score_threshold = 0.5;

        for (i, stride) in strides.iter().enumerate() {
            let cls = &outputs[i];
            let obj = &outputs[i + strides.len() * 1];
            let bbox = &outputs[i + strides.len() * 2];
            let kkps = &outputs[i + strides.len() * 3];

            let rows: usize = (sizes.0 as usize / stride) as usize;
            let cols = (sizes.1 as usize / stride) as usize;

            for row in 0..rows {
                for col in 0..cols {
                    let idx = row * cols + col;

                    let cls_score = cls.clone().slice([0, idx, 0]).into_scalar();
                    let obj_score = obj.clone().slice([0, idx, 0]).into_scalar();

                    let cls_score = cls_score.min(1.0).max(0.0);
                    let obj_score = obj_score.min(1.0).max(0.0);

                    let score = f32::sqrt(cls_score * obj_score);
                    // Check if the score meets the threshold
                    if score < score_threshold {
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

        // TODO Let user configure nms threshold
        non_maximum_suppression(&mut boxes, &mut lms, 0.3);
        (boxes, lms)
    }
}

impl Detector for Yunet {
    fn detect(&self, input: &DynamicImage) -> Vec<FacialAreaRegion> {
        let sizes = resize_to_multiple_of_divisor(input.width(), input.height(), 32, Some(640));
        let resized = input
            .resize_exact(sizes.1, sizes.0, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        let (width, height) = (sizes.1, sizes.0);

        let device = Default::default();
        // Create tensor from image data
        let x = to_tensor(
            resized.into_raw(),
            [height as usize, width as usize, 3],
            &device,
        )
        .unsqueeze::<4>(); // [B, C, H, W]

        let outputs = self.model.forward(x).to_vec();

        let (detections, lms) = self.postprocess(outputs, sizes);

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
                confidence: Some(confidence)
            };
            results.push(facial_area);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use crate::detection::{Detector, Yunet};

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");

        let model = Yunet::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(&img);

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
