use burn::{
    tensor::{Int, Tensor, TensorData},
    vision::{Nms, NmsOptions},
};

use super::{
    resize_tensor, BoundingBox, Detector, DetectorMetadata,
    FacialAreaRegion, Landmarks, ResizedDimensions,
};
use crate::ImageToTensor;

mod centerface {
    include!(concat!(env!("OUT_DIR"), "/models/detection/centerface.rs"));
}

/// Centerface face detector.
///
/// Model and resources: [Star-Clouds – CenterFace](https://github.com/Star-Clouds/CenterFace)
///
/// Licensed under the [MIT License](https://github.com/Star-Clouds/CenterFace/blob/master/LICENSE).
///
/// # Reference:
///
/// ```text
/// @inproceedings{CenterFace,
///   title   = {CenterFace: Joint Face Detection and Alignment Using Face as Point},
///   author  = {Xu, Yuanyuan and Yan, Wan and Sun, Haixin and Yang, Genke and Luo, Jiliang},
///   booktitle = {arXiv:1911.03599},
///   year    = {2019}
/// }
/// ```
pub struct CenterFace {
    model: centerface::Model,
}

impl CenterFace {
    // Create a new Centerface face detector
    pub fn new() -> Self {
        let model = centerface::Model::default();
        Self { model: model }
    }

    fn postprocess(
        &self,
        heatmap: Tensor<4>,
        landmark: Tensor<4>,
        offset: Tensor<4>,
        scale: Tensor<4>,
        sizes: ResizedDimensions,
        nms_options: NmsOptions,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let device = heatmap.device();

        // Dimensions
        let h = heatmap.dims()[2];
        let w = heatmap.dims()[3];
        let hw = h * w;

        // Flatten spatial dimensions [1, C, H, W] -> [C, H*W]
        // (Assuming batch size of 1)
        let heatmap_1d = heatmap.reshape([hw]);
        let scale_flat = scale.reshape([2, hw]);
        let offset_flat = offset.reshape([2, hw]);
        let landmark_flat = landmark.reshape([10, hw]);

        let idx = Tensor::<1, Int>::arange(0..hw as i64, &device);

        let row_idx = idx.clone() / (w as i64);
        let col_idx = idx - (row_idx.clone() * (w as i64));

        let grid_y = row_idx.float().reshape([1, hw]);
        let grid_x = col_idx.float().reshape([1, hw]);

        // scale0 corresponds to height, scale1 corresponds to width
        let s0 = scale_flat.clone().slice([0..1, 0..hw]).exp() * 4.0;
        let s1 = scale_flat.clone().slice([1..2, 0..hw]).exp() * 4.0;

        let o0 = offset_flat.clone().slice([0..1, 0..hw]);
        let o1 = offset_flat.clone().slice([1..2, 0..hw]);

        let w_f32 = sizes.width as f32;
        let h_f32 = sizes.height as f32;

        let x1_unclamped = (grid_x.clone() + o1 + 0.5) * 4.0 - s1.clone() / 2.0;
        let y1_unclamped = (grid_y.clone() + o0 + 0.5) * 4.0 - s0.clone() / 2.0;

        let x1 = x1_unclamped.clamp(0.0, w_f32);
        let y1 = y1_unclamped.clamp(0.0, h_f32);

        let x2 = (x1.clone() + s1.clone()).clamp(0.0, w_f32);
        let y2 = (y1.clone() + s0.clone()).clamp(0.0, h_f32);

        // Cat vertically into [4, H*W], then transpose into [H*W, 4] for NMS
        let bboxes =
            Tensor::cat(vec![x1.clone(), y1.clone(), x2.clone(), y2.clone()], 0).transpose();

        let mut lm_tensors = Vec::with_capacity(10);
        for j in 0..5 {
            // lm0 (even indexes) = Y ratio offset
            // lm1 (odd indexes) = X ratio offset
            let lm0 = landmark_flat.clone().slice([j * 2..j * 2 + 1, 0..hw]);
            let lm1 = landmark_flat.clone().slice([j * 2 + 1..j * 2 + 2, 0..hw]);

            let lm_x = lm1 * s1.clone() + x1.clone();
            let lm_y = lm0 * s0.clone() + y1.clone();

            lm_tensors.push(lm_x);
            lm_tensors.push(lm_y);
        }

        // Cat vertically into [10, H*W], then transpose into [H*W, 10]
        let lms = Tensor::cat(lm_tensors, 0).transpose();

        let kept_indices = bboxes.clone().nms(heatmap_1d.clone(), nms_options);

        if kept_indices.dims()[0] == 0 {
            return (Vec::new(), Vec::new());
        }

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

        // Filter out bad anchors and scale natively using division math mapping
        let kept_bboxes = bboxes.select(0, kept_indices.clone()) / scale_bbox;
        let kept_lms = lms.select(0, kept_indices.clone()) / scale_lms;
        let kept_scores = heatmap_1d.select(0, kept_indices);

        let final_bboxes_data = kept_bboxes
            .into_data()
            .to_vec::<f32>()
            .expect("Read bboxes");
        let final_scores_data = kept_scores
            .into_data()
            .to_vec::<f32>()
            .expect("Read scores");
        let final_lms_data = kept_lms
            .into_data()
            .to_vec::<f32>()
            .expect("Read landmarks");

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

impl DetectorMetadata for CenterFace {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = None;
}

impl Detector for CenterFace {
    /// See [`super::Detector`]
    fn detect<I: ImageToTensor>(
        &self,
        input: &I,
        nms_options: NmsOptions,
    ) -> Vec<FacialAreaRegion> {
        let (tensor, sizes) = resize_tensor(input.to_tensor(), Self::DIVISOR, Self::MAX_SIZE);

        let (heatmap, scale, offset, lms) = self.model.forward(tensor);

        let (detections, lms) = self.postprocess(heatmap, lms, offset, scale, sizes, nms_options);

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
                landmarks: Some(*landmark),
            };
            results.push(facial_area);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use crate::detection::{CenterFace, Detector, NmsOptions};

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");

        let model: CenterFace = CenterFace::new();

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
