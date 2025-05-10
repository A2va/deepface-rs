use burn::{backend::ndarray::NdArray, tensor::Tensor};
use image::DynamicImage;
use tuple_conv::RepeatedTuple;

use super::{Detector, FacialAreaRegion};
use super::{resize_to_multiple_of_divisor, to_tensor};

mod yunet {
    include!(concat!(env!("OUT_DIR"), "/models/detection/yunet.rs"));
}

// https://github.com/opencv/opencv/blob/829495355d7da3f073828dd584f1cdba9e07dc65/modules/objdetect/src/face_detect.cpp#L20

type DeepFaceBackend = NdArray<f32>;

/// Yunet face detector.
///
/// A lightweight fast face detection model trained by OpenCV.  
/// It predicts face bounding boxes and landmarks from an input image.  
///
/// Model and resources: [OpenCV Zoo – Yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)  
///
/// # Licensing
/// - Model weights: [MIT License](https://opensource.org/licenses/MIT)  
/// - OpenCV reference implementation: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)  
pub struct Yunet {
    model: yunet::Model<DeepFaceBackend>,
}

impl Yunet {
    /// Create a new Yunet face detector.
    pub fn new() -> Self {
        let model = yunet::Model::default();
        Self { model }
    }

    fn postprocess(&self, outputs: Vec<Tensor<DeepFaceBackend, 3>>, sizes: (u32, u32, f32, f32))  -> (Vec<[f32; 4]>, Vec<f32>, Vec<[(f32, f32); 5]>){
        let (mut dets, scores, mut lms) = self.decode(outputs, sizes);

        let (height, width, scale_h, scale_w) = sizes;

        if !dets.is_empty() {
            dets = dets
                .into_iter()
                .map(|[x1, y1, x2, y2]| [x1 / scale_w, y1 / scale_h, x2 / scale_w, y2 / scale_h])
                .collect::<Vec<[f32; 4]>>();

            // Scale landmarks
            lms = lms
                .into_iter()
                .map(|mut landmark| {
                    for i in 0..5 {
                        landmark[i] = (landmark[i].0 / scale_w, landmark[i].1 / scale_h)
                    }
                    landmark
                })
                .collect::<Vec<[(f32, f32); 5]>>();
        }
        (dets, scores, lms)
    }

    fn decode(&self, outputs: Vec<Tensor<DeepFaceBackend, 3>>, sizes: (u32, u32, f32, f32)) -> (Vec<[f32; 4]>, Vec<f32>, Vec<[(f32, f32); 5]>) {
        let strides = [8, 16, 32];

        let mut scores = Vec::new();
        let mut boxes: Vec<[f32; 4]> = Vec::new();
        let mut lms = Vec::new();

        let score_threshold = 0.8;

        for (i, stride) in strides.iter().enumerate() {
            let cls = &outputs[i];
            let obj = &outputs[i + strides.len() * 1];
            let bbox = &outputs[i + strides.len() * 2];
            let kkps = &outputs[i + strides.len() * 3];

            let rows = (sizes.0 as usize / stride) as usize;
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

                    scores.push(score);

                    let cx =
                        (col as f32 + bbox.clone().slice([0, idx, 0]).into_scalar()) * (*stride as f32);
                    let cy =
                        (row as f32 + bbox.clone().slice([0, idx, 1]).into_scalar()) * (*stride as f32);
                    let w = f32::exp(bbox.clone().slice([0, idx, 2]).into_scalar()) * (*stride as f32);
                    let h = f32::exp(bbox.clone().slice([0, idx, 3]).into_scalar()) * (*stride as f32);

                    let x1 = cx - w / 2.0;
                    let y1 = cy - h / 2.0;

                    let x2 = cx + w / 2.0;
                    let y2 = cy + h / 2.0;
                    boxes.push([x1, y1, x2, y2]);

                
                    let mut lm: [(f32, f32); 5] = [(0.0, 0.0); 5];
                    // Get landmarks
                    for n in 0..5 {
                        let landmark_x = (kkps.clone().slice([0, idx, n * 2]).into_scalar() + col as f32) * (*stride as f32);
                        let landmark_y = (kkps.clone().slice([0, idx, n * 2 + 1]).into_scalar() + row as f32) * (*stride as f32);
                        
                        lm[n] = (landmark_x, landmark_y);
                    }
                    lms.push(lm);
                }
            }
        }

        let keep: Vec<usize> = self.nms(&boxes, &scores, 0.3);

        // Keep only detections at indices in `keep`
        boxes = keep.iter().map(|&i| boxes[i]).collect::<Vec<[f32; 4]>>();
            lms = keep
                .iter()
                .map(|&i| lms[i].clone())
                .collect::<Vec<[(f32, f32); 5]>>();
            scores = keep.iter().map(|&i| scores[i]).collect::<Vec<f32>>();

        (boxes, scores, lms)
    }

    fn nms(&self, boxes: &Vec<[f32; 4]>, scores: &Vec<f32>, nms_thresh: f32) -> Vec<usize> {
        let num_detections = boxes.len();

        // Sort indices by score descending
        let mut indices: Vec<usize> = (0..num_detections).collect();
        indices.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());

        let mut suppressed = vec![false; num_detections];
        let mut keep = Vec::new();

        // Precompute areas
        let areas: Vec<f32> = boxes
            .iter()
            .map(|b| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0))
            .collect();

        for _i in 0..num_detections {
            let i = indices[_i];
            if suppressed[i] {
                continue;
            }
            keep.push(i);

            let (ix1, iy1, ix2, iy2) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
            let iarea = areas[i];

            for _j in (_i + 1)..num_detections {
                let j = indices[_j];
                if suppressed[j] {
                    continue;
                }

                let (xx1, yy1) = (ix1.max(boxes[j][0]), iy1.max(boxes[j][1]));
                let (xx2, yy2) = (ix2.min(boxes[j][2]), iy2.min(boxes[j][3]));

                let w = (xx2 - xx1 + 1.0).max(0.0);
                let h = (yy2 - yy1 + 1.0).max(0.0);

                let inter = w * h;
                let ovr = inter / (iarea + areas[j] - inter);
                if ovr >= nms_thresh {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }
}

impl Detector for Yunet {
    fn detect(&self, input: &DynamicImage) -> Vec<FacialAreaRegion> {
        let sizes = resize_to_multiple_of_divisor(input.width(), input.height(), 32);
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

       let (detections, scores, lms) =
            self.postprocess(outputs, sizes);

        let mut results = Vec::new();
        for (i, detection) in detections.iter().enumerate() {
            let x = detection[0];
            let y = detection[1];
            let w = detection[2] - x;
            let h = detection[3] - y;

            let landmark = &lms[i];

            let right_eye = (landmark[0].0 as u32, landmark[0].1 as u32);
            let left_eye = (landmark[1].0 as u32, landmark[1].1 as u32);

            let nose = (landmark[2].0 as u32, landmark[2].1 as u32);
            let right_mouth = (landmark[3].0 as u32, landmark[3].1 as u32);
            let left_mouth = (landmark[4].0 as u32, landmark[4].1 as u32);

            let score = f32::max(scores[i], 0.0);
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
                confidence: Some(f32::min(score, 1.0)),
            };
            results.push(facial_area);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_line() {
        let p = std::env::current_dir().unwrap();
        println!("{}", p.display());
    }
}