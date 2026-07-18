use burn::{
    tensor::{Int, Tensor, TensorData},
    vision::{Nms, NmsOptions},
};
use tuple_conv::RepeatedTuple;

use super::{
    resize_tensor, BoundingBox, Detector, DetectorMetadata,
    FacialAreaRegion, Landmarks, ResizedDimensions,
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
pub struct Yunet {
    model: yunet::Model,
}

impl Yunet {
    /// Create a new Yunet face detector.
    pub fn new() -> Self {
        let model = yunet::Model::default();
        Self { model }
    }

    // Original implementation
    // https://github.com/opencv/opencv/blob/829495355d7da3f073828dd584f1cdba9e07dc65/modules/objdetect/src/face_detect.cpp
    fn postprocess(
        &self,
        outputs: Vec<Tensor<3>>,
        sizes: ResizedDimensions,
        nms_options: NmsOptions,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let device = outputs[0].device();
        let strides = [8, 16, 32];

        let mut bboxes_list = Vec::with_capacity(3);
        let mut scores_list = Vec::with_capacity(3);
        let mut lms_list = Vec::with_capacity(3);

        for (i, stride) in strides.iter().enumerate() {
            // Explicitly limit total elements to original loop sizes to avoid processing padding!
            let rows = (sizes.height as usize / stride) as usize;
            let cols = (sizes.width as usize / stride) as usize;
            let valid_n = rows * cols;

            // Slice out any extra padded anchors the model might have emitted
            let cls = outputs[i].clone().slice([0..1, 0..valid_n, 0..1]);
            let obj = outputs[i + strides.len() * 1]
                .clone()
                .slice([0..1, 0..valid_n, 0..1]);
            let bbox = outputs[i + strides.len() * 2]
                .clone()
                .slice([0..1, 0..valid_n, 0..4]);
            let kkps = outputs[i + strides.len() * 3]
                .clone()
                .slice([0..1, 0..valid_n, 0..10]);

            let cols_i64 = cols as i64;

            let idx = Tensor::<1, Int>::arange(0..valid_n as i64, &device);

            let row_idx = idx.clone() / cols_i64;
            let col_idx = idx - (row_idx.clone() * cols_i64);

            let col_float = col_idx.float().reshape([1, valid_n, 1]);
            let row_float = row_idx.float().reshape([1, valid_n, 1]);

            let grid_tensor = Tensor::cat(vec![col_float, row_float], 2);

            let cls_clamped = cls.clamp(0.0, 1.0);
            let obj_clamped = obj.clamp(0.0, 1.0);
            let scores = (cls_clamped * obj_clamped).sqrt();

            let bbox_xy = bbox.clone().slice([0..1, 0..valid_n, 0..2]);
            let bbox_wh = bbox.clone().slice([0..1, 0..valid_n, 2..4]);

            let stride_f32 = *stride as f32;
            let cxcy = (grid_tensor.clone() + bbox_xy) * stride_f32;
            let wh = bbox_wh.exp() * stride_f32;

            let half_wh = wh / 2.0;
            let x1y1 = cxcy.clone() - half_wh.clone();
            let x2y2 = cxcy + half_wh;

            let bboxes = Tensor::cat(vec![x1y1, x2y2], 2);

            let grid_10 = Tensor::cat(
                vec![
                    grid_tensor.clone(),
                    grid_tensor.clone(),
                    grid_tensor.clone(),
                    grid_tensor.clone(),
                    grid_tensor.clone(),
                ],
                2,
            );
            let lms = (kkps + grid_10) * stride_f32;

            bboxes_list.push(bboxes);
            scores_list.push(scores);
            lms_list.push(lms);
        }

        let all_bboxes = Tensor::cat(bboxes_list, 1);
        let all_scores = Tensor::cat(scores_list, 1);
        let all_lms = Tensor::cat(lms_list, 1);

        let total_n = all_bboxes.dims()[1];
        let bboxes_2d = all_bboxes.reshape([total_n, 4]);
        let scores_1d = all_scores.reshape([total_n]);
        let lms_2d = all_lms.reshape([total_n, 10]);

        let kept_indices = bboxes_2d.clone().nms(scores_1d.clone(), nms_options);

        let scale_bbox = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    sizes.width_scale,
                    sizes.height_scale,
                    sizes.width_scale,
                    sizes.height_scale,
                ],
                [1, 4],
            ),
            &device,
        );
        let scale_lms = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    sizes.width_scale,
                    sizes.height_scale,
                    sizes.width_scale,
                    sizes.height_scale,
                    sizes.width_scale,
                    sizes.height_scale,
                    sizes.width_scale,
                    sizes.height_scale,
                    sizes.width_scale,
                    sizes.height_scale,
                ],
                [1, 10],
            ),
            &device,
        );

        let kept_bboxes = bboxes_2d.select(0, kept_indices.clone()) / scale_bbox;
        let kept_lms = lms_2d.select(0, kept_indices.clone()) / scale_lms;
        let kept_scores = scores_1d.select(0, kept_indices);

        let final_bboxes_data = kept_bboxes
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read bboxes");
        let final_scores_data = kept_scores
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read scores");
        let final_lms_data = kept_lms
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to read landmarks");

        let num_kept = final_scores_data.len();
        let mut final_boxes = Vec::with_capacity(num_kept);
        let mut final_landmarks = Vec::with_capacity(num_kept);

        for i in 0..num_kept {
            final_boxes.push(BoundingBox {
                xmin: final_bboxes_data[i * 4 + 0],
                ymin: final_bboxes_data[i * 4 + 1],
                xmax: final_bboxes_data[i * 4 + 2],
                ymax: final_bboxes_data[i * 4 + 3],
                confidence: final_scores_data[i],
            });

            let mut lm: Landmarks = [(0.0, 0.0); 5];
            for n in 0..5 {
                lm[n] = (
                    final_lms_data[i * 10 + n * 2],
                    final_lms_data[i * 10 + n * 2 + 1],
                );
            }
            final_landmarks.push(lm);
        }

        (final_boxes, final_landmarks)
    }
}

impl DetectorMetadata for Yunet {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = Some(640);
}

impl Detector for Yunet {
    /// See [`super::Detector`]
    fn detect<I: ImageToTensor>(
        &self,
        input: &I,
        nms_options: NmsOptions,
    ) -> Vec<FacialAreaRegion> {
        let (tensor, sizes) = resize_tensor(input.to_tensor(), Self::DIVISOR, Self::MAX_SIZE);

        let outputs = self.model.forward(tensor).to_vec();

        let (detections, lms) = self.postprocess(outputs, sizes, nms_options);

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

            let confidence = detection.confidence.clamp(0.0, 1.0);
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
                landmarks: Some(*landmark),
            };
            results.push(facial_area);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use crate::detection::{Detector, NmsOptions, Yunet};

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");
        let model: Yunet = Yunet::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(
            &img,
            NmsOptions {
                score_threshold: 0.8,
                ..NmsOptions::default()
            },
        );

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
