use crate::detection::{Detector, FacialAreaRegion};
use burn::tensor::backend::Backend;
use burn::{backend::ndarray::NdArray, tensor::Tensor};
use image::DynamicImage;

use super::{non_maximum_suppression, to_tensor, BoundingBox, DeepFaceBackend, Landmarks};

mod centerface {
    include!(concat!(env!("OUT_DIR"), "/models/detection/centerface.rs"));
}

/// Centerface face detector.
///
/// Model and resources: [Star-Clouds – CenterFace](https://github.com/Star-Clouds/CenterFace)    
///
/// Licensed under the [MIT License](https://opensource.org/licenses/MIT).
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
    model: centerface::Model<DeepFaceBackend>,
}

impl CenterFace {
    // Create a new Centerface face detector
    pub fn new() -> Self {
        let model = centerface::Model::default();
        Self { model: model }
    }

    fn transform(&self, width: u32, height: u32) -> (u32, u32, f32, f32) {
        let width = width as f32;
        let height = height as f32;

        let new_height = f32::ceil(height / 32.0) * 32.0;
        let new_width = f32::ceil(width / 32.0) * 32.0;

        let scale_width = new_width / width as f32;
        let scale_height = new_height / height as f32;

        (
            new_height as u32,
            new_width as u32,
            scale_height,
            scale_width,
        )
    }

    fn postprocess(
        &self,
        heatmap: Tensor<DeepFaceBackend, 4>,
        landmark: Tensor<DeepFaceBackend, 4>,
        offset: Tensor<DeepFaceBackend, 4>,
        scale: Tensor<DeepFaceBackend, 4>,
        transformed_size: (u32, u32, f32, f32),
        threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let (height, width, scale_h, scale_w) = transformed_size;

        let (mut dets, mut lms) =
            self.decode(heatmap, scale, offset, landmark, (height, width), threshold);

        if !dets.is_empty() {
            // Scale dets
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
        heatmap: Tensor<DeepFaceBackend, 4>,
        scale: Tensor<DeepFaceBackend, 4>,
        offset: Tensor<DeepFaceBackend, 4>,
        landmark: Tensor<DeepFaceBackend, 4>,
        size: (u32, u32),
        threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        // np.squeeze remove all dims that have a size of 1, but it will not work with burn
        // since I know only the dim 1 of the heapmap is 1 I will use squeeze on the dim 1
        let heatmap = heatmap.squeeze_dims::<2>(&[0, 1]);

        let scale_dim2 = scale.dims()[2];
        let scale_dim3 = scale.dims()[3];

        let scale0: Tensor<DeepFaceBackend, 2> = scale
            .clone()
            .slice([0..1, 0..1])
            .reshape([scale_dim2, scale_dim3]);
        let scale1: Tensor<DeepFaceBackend, 2> = scale
            .clone()
            .slice([0..1, 1..2])
            .reshape([scale_dim2, scale_dim3]);

        let offset_dim2 = offset.dims()[2];
        let offset_dim3 = offset.dims()[3];

        let offset0 = offset
            .clone()
            .slice([0..1, 0..1])
            .reshape([offset_dim2, offset_dim3]);
        let offset1 = offset
            .clone()
            .slice([0..1, 1..2])
            .reshape([offset_dim2, offset_dim3]);

        let t = heatmap.clone().greater_elem(threshold).nonzero();
        let c0: Vec<u32> = t[0]
            .clone()
            .into_data()
            .convert_dtype(burn::tensor::DType::U32)
            .to_vec()
            .unwrap();
        let c1: Vec<u32> = t[1]
            .clone()
            .into_data()
            .convert_dtype(burn::tensor::DType::U32)
            .to_vec()
            .unwrap();

        let mut boxes = Vec::new();
        let mut lms = Vec::new();

        if !c0.is_empty() {
            for i in 0..c0.len() {
                let ci0 = c0[i] as usize;
                let ci1 = c1[i] as usize;

                let s0: f32 = scale0
                    .clone()
                    .slice([ci0, ci1])
                    .exp()
                    .mul_scalar(4.0)
                    .into_scalar();
                let s1 = scale1
                    .clone()
                    .slice([ci0, ci1])
                    .exp()
                    .mul_scalar(4.0)
                    .into_scalar();

                let o0 = offset0.clone().slice([ci0, ci1]);
                let o1 = offset1.clone().slice([ci0, ci1]);

                let score = heatmap.clone().slice([ci0, ci1]);

                let mut x1 = f32::max(0.0, (ci1 as f32 + o1.into_scalar() + 0.5) * 4.0 - s1 / 2.0);
                let mut y1 = f32::max(0.0, (ci0 as f32 + o0.into_scalar() + 0.5) * 4.0 - s0 / 2.0);

                x1 = f32::min(x1, size.1 as f32);
                y1 = f32::min(y1, size.0 as f32);

                let x2 = f32::min(x1 + s1, size.1 as f32);
                let y2 = f32::min(y1 + s0, size.0 as f32);

                boxes.push(BoundingBox {
                    xmin: x1,
                    ymin: y1,
                    xmax: x2,
                    ymax: y2,
                    confidence: score.into_scalar(),
                });

                let mut lm: Landmarks = [(0.0, 0.0); 5];
                for j in 0..5 {
                    let lm0 = landmark.clone().slice([0, j * 2, ci0, ci1]).into_scalar();
                    let lm1 = landmark
                        .clone()
                        .slice([0, j * 2 + 1, ci0, ci1])
                        .into_scalar();

                    lm[j] = (lm1 * s1 + x1, lm0 * s0 + y1);
                }
                lms.push(lm);
            }
            non_maximum_suppression(&mut boxes, &mut lms, 0.3);
        }
        (boxes, lms)
    }
}

impl Detector for CenterFace {
    fn detect(&self, input: &DynamicImage) -> Vec<FacialAreaRegion> {
        let device = <DeepFaceBackend as burn::tensor::backend::Backend>::Device::default();
        let threshold = 0.7;

        // Resize the input image and conver it to a float vector
        let sizes = self.transform(input.width(), input.height());
        let resized = input
            .resize_exact(sizes.1, sizes.0, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        let device = Default::default();
        // Create tensor from image data
        let x = to_tensor(
            resized.into_raw(),
            [sizes.0 as usize, sizes.1 as usize, 3],
            &device,
        )
        .unsqueeze::<4>(); // [B, C, H, W]

        let (heatmap, scale, offset, lms) = self.model.forward(x);

        let (detections, lms) = self.postprocess(heatmap, lms, offset, scale, sizes, threshold);

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
    use crate::detection::{CenterFace, Detector};

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");

        let model = CenterFace::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(&img);

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
