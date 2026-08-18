use burn::{
    tensor::{Int, Tensor, TensorData},
    vision::{Nms, NmsOptions},
};

use super::{
    resize_tensor, BoundingBox, Detector, DetectorMetadata, FacialAreaRegion, Landmarks,
    ResizedDimensions,
};
use crate::ImageToTensor;
use crate::landmarks::ModelKind;

#[cfg(feature = "scrfd-10g")]
mod scrfd_10g {
    include!(concat!(env!("OUT_DIR"), "/models/detection/scrfd_10g.rs"));
}

#[cfg(feature = "scrfd-500m")]
mod scrfd_500m {
    include!(concat!(env!("OUT_DIR"), "/models/detection/scrfd_500m.rs"));
}

/// Type alias for SCRFD output tuple:
/// (s8, s16, s32, b8, b16, b32, k8, k16, k32)
type ScrfdOutput = (
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
    Tensor<2>,
);

/// Abstraction trait over Burn-generated SCRFD ONNX models.
trait ScrfdModel: Send + Sync {
    fn forward(&self, input: Tensor<4>) -> ScrfdOutput;
}

#[cfg(feature = "scrfd-10g")]
impl ScrfdModel for scrfd_10g::Model {
    fn forward(&self, input: Tensor<4>) -> ScrfdOutput {
        self.forward(input)
    }
}

#[cfg(feature = "scrfd-500m")]
impl ScrfdModel for scrfd_500m::Model {
    fn forward(&self, input: Tensor<4>) -> ScrfdOutput {
        self.forward(input)
    }
}

/// Generic SCRFD face detector wrapping any SCRFD model architecture.
pub struct Scrfd<M> {
    model: M,
}

impl<M: Default> Scrfd<M> {
    /// Create a detector using default model weights.
    pub fn new() -> Self {
        Self {
            model: M::default(),
        }
    }
}

/// SCRFD 10G face detector.
///
/// Efficient face detection model based on "Sample and Computation Redistribution
/// for Efficient Face Detection" (SCRFD).
///
/// Model and resources: [InsightFace – SCRFD](https://github.com/insightface/insightface)
///
/// # Licensing
/// - Model weights: [non-commercial academic research only](https://www.insightface.ai/#pricing)
/// - Code: [Apache 2.0 License](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/LICENSE)
///
/// # Reference:
/// ```text
/// @article{guo2021sample,
///   title={Sample and computation redistribution for efficient face detection},
///   author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
///   journal={arXiv preprint arXiv:2105.04714},
///   year={2021}
/// }scrfd_10g
/// ```
#[cfg(feature = "scrfd-10g")]
pub type Scrfd10g = Scrfd<scrfd_10g::Model>;

/// SCRFD 500m face detector.
///
/// Efficient face detection model based on "Sample and Computation Redistribution
/// for Efficient Face Detection" (SCRFD).
///
/// Model and resources: [InsightFace – SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
///
/// # Licensing
/// - Model weights: [non-commercial academic research only](https://www.insightface.ai/#pricing)
/// - Code: [Apache 2.0 License](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/LICENSE)
///
/// # Reference:
/// ```text
/// @article{guo2021sample,
///   title={Sample and computation redistribution for efficient face detection},
///   author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
///   journal={arXiv preprint arXiv:2105.04714},
///   year={2021}
/// }scrfd_10g
/// ```
#[cfg(feature = "scrfd-500m")]
pub type Scrfd500m = Scrfd<scrfd_500m::Model>;

impl<M> DetectorMetadata for Scrfd<M> {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = Some(640);
}

impl<M: ScrfdModel> Detector for Scrfd<M> {
    fn detect<I: ImageToTensor>(
        &self,
        input: &I,
        nms_options: NmsOptions,
    ) -> Vec<FacialAreaRegion> {
        let (tensor, sizes) = resize_tensor(input.to_tensor(), Self::DIVISOR, Self::MAX_SIZE);

        // Normalize image tensor for SCRFD: (image - 127.5) / 127.5
        let tensor = (tensor - 127.5) / 127.5;

        // Execute inference (passed by value without unnecessary clone)
        let outputs = self.model.forward(tensor);

        let (detections, lms) = postprocess(outputs, sizes, nms_options);

        let mut results = Vec::with_capacity(detections.len());
        for (i, detection) in detections.iter().enumerate() {
            let x = detection.xmin;
            let y = detection.ymin;
            let w = detection.xmax - x;
            let h = detection.ymax - y;

            let landmark = &lms[i];
            let confidence = detection.confidence.clamp(0.0, 1.0);
            let facial_area = FacialAreaRegion {
                x: x as u32,
                y: y as u32,
                w: w as u32,
                h: h as u32,
                confidence: Some(confidence),
                landmarks: Some(landmark.clone()),
            };
            results.push(facial_area);
        }

        results
    }
}

// Based on https://github.com/yakhyo/uniface/blob/main/uniface/detection/scrfd.py
#[allow(clippy::type_complexity)]
fn postprocess(
    outputs: ScrfdOutput,
    sizes: ResizedDimensions,
    nms_options: NmsOptions,
) -> (Vec<BoundingBox>, Vec<Landmarks>) {
    let (s8, s16, s32, b8, b16, b32, k8, k16, k32) = outputs;
    let device = s8.device();

    let strides = [8, 16, 32];
    let score_outputs = [s8, s16, s32];
    let bbox_outputs = [b8, b16, b32];
    let kps_outputs = [k8, k16, k32];

    // SCRFD uses 2 anchors per spatial location across all feature map levels
    let num_anchors = 2i64;

    let mut bboxes_list = Vec::with_capacity(3);
    let mut scores_list = Vec::with_capacity(3);
    let mut lms_list = Vec::with_capacity(3);

    for (i, &stride) in strides.iter().enumerate() {
        let stride_f32 = stride as f32;
        let rows = (sizes.height as usize / stride) as usize;
        let cols = (sizes.width as usize / stride) as usize;
        let valid_n = rows * cols * (num_anchors as usize);

        // Slice out valid spatial predictions (removing any padded output elements)
        let score = score_outputs[i].clone().slice([0..valid_n, 0..1]);
        let bbox_pred = bbox_outputs[i].clone().slice([0..valid_n, 0..4]) * stride_f32;
        let kps_pred = kps_outputs[i].clone().slice([0..valid_n, 0..10]) * stride_f32;

        // Generate anchor centers for this feature map level
        let cols_i64 = cols as i64;
        let idx = Tensor::<1, Int>::arange(0..valid_n as i64, &device);
        let cell_idx = idx / num_anchors;
        let row_idx = cell_idx.clone() / cols_i64;
        let col_idx = cell_idx - (row_idx.clone() * cols_i64);

        let col_float = col_idx.float().reshape([valid_n, 1]);
        let row_float = row_idx.float().reshape([valid_n, 1]);

        // Spatial anchor coordinates in pixel space
        let anchor_centers = Tensor::cat(vec![col_float, row_float], 1) * stride_f32;

        // Decode bounding boxes
        let anchor_x = anchor_centers.clone().slice([0..valid_n, 0..1]);
        let anchor_y = anchor_centers.clone().slice([0..valid_n, 1..2]);

        let dist_l = bbox_pred.clone().slice([0..valid_n, 0..1]);
        let dist_t = bbox_pred.clone().slice([0..valid_n, 1..2]);
        let dist_r = bbox_pred.clone().slice([0..valid_n, 2..3]);
        let dist_b = bbox_pred.clone().slice([0..valid_n, 3..4]);

        let x1 = anchor_x.clone() - dist_l;
        let y1 = anchor_y.clone() - dist_t;
        let x2 = anchor_x.clone() + dist_r;
        let y2 = anchor_y.clone() + dist_b;

        let bboxes = Tensor::cat(vec![x1, y1, x2, y2], 1);

        // Decode 5 keypoints
        // Concatenate anchor_centers 5 times to form [x, y, x, y, x, y, x, y, x, y]
        let anchor_kps = Tensor::cat(vec![anchor_centers; 5], 1);
        let lms = anchor_kps + kps_pred;

        bboxes_list.push(bboxes);
        scores_list.push(score);
        lms_list.push(lms);
    }

    // Concatenate all feature map levels
    let all_bboxes = Tensor::cat(bboxes_list, 0);
    let all_scores = Tensor::cat(scores_list, 0);
    let all_lms = Tensor::cat(lms_list, 0);

    let total_n = all_bboxes.dims()[0];
    let bboxes_2d = all_bboxes.reshape([total_n, 4]);
    let scores_1d = all_scores.reshape([total_n]);
    let lms_2d = all_lms.reshape([total_n, 10]);

    // Non-Maximum Suppression
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

        let points: Vec<(f32, f32)> = (0..5)
            .map(|n| {
                (
                    final_lms_data[i * 10 + n * 2],
                    final_lms_data[i * 10 + n * 2 + 1],
                )
            })
            .collect();
        final_landmarks.push(Landmarks::new(points, ModelKind::Five));
    }

    (final_boxes, final_landmarks)
}

#[cfg(test)]
mod tests {
    use crate::detection::{Detector, NmsOptions, Scrfd10g};

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");
        let model: Scrfd10g = Scrfd10g::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(
            &img,
            NmsOptions {
                score_threshold: 0.5,
                ..NmsOptions::default()
            },
        );

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
