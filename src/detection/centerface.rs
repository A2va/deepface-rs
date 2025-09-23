use burn::{prelude::Backend, tensor::Tensor};

use super::{
    non_maximum_suppression, resize_tensor, BoundingBox, Detector, FacialAreaRegion,
    Landmarks, ResizedDimensions,
};
use crate::ImageToTensor;

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
pub struct CenterFace<B: Backend> {
    model: centerface::Model<B>,
}

impl<B: Backend<FloatElem = f32>> CenterFace<B> {
    // Construct a new instance of the model with a specific burn backend
    // Create a new Centerface face detector
    pub fn new() -> Self {
        let model = centerface::Model::default();
        Self { model: model }
    }

    fn postprocess(
        &self,
        heatmap: Tensor<B, 4>,
        landmark: Tensor<B, 4>,
        offset: Tensor<B, 4>,
        scale: Tensor<B, 4>,
        sizes: ResizedDimensions,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        let (mut dets, mut lms) = self.decode(
            heatmap,
            scale,
            offset,
            landmark,
            sizes,
            confidence_threshold,
            nms_threshold,
        );

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
        heatmap: Tensor<B, 4>,
        scale: Tensor<B, 4>,
        offset: Tensor<B, 4>,
        landmark: Tensor<B, 4>,
        sizes: ResizedDimensions,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> (Vec<BoundingBox>, Vec<Landmarks>) {
        // np.squeeze remove all dims that have a size of 1, but it will not work with burn
        // since I know only the dim 1 of the heapmap is 1 I will use squeeze on the dim 1
        let heatmap = heatmap.squeeze_dims::<2>(&[0, 1]);

        let scale_dim2 = scale.dims()[2];
        let scale_dim3 = scale.dims()[3];

        let scale0: Tensor<B, 2> = scale
            .clone()
            .slice([0..1, 0..1])
            .reshape([scale_dim2, scale_dim3]);
        let scale1: Tensor<B, 2> = scale
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

        let t = heatmap.clone().greater_elem(confidence_threshold).nonzero();
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

                let s0 = scale0
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

                x1 = f32::min(x1, sizes.width as f32);
                y1 = f32::min(y1, sizes.height as f32);

                let x2 = f32::min(x1 + s1, sizes.width as f32);
                let y2 = f32::min(y1 + s0, sizes.height as f32);

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
            non_maximum_suppression(&mut boxes, &mut lms, nms_threshold);
        }
        (boxes, lms)
    }
}

impl<B: Backend<FloatElem = f32>> Detector<B> for CenterFace<B> {
    const DIVISOR: u32 = 32;
    const MAX_SIZE: Option<u32> = None;

    /// See [`super::Detector`]
    fn detect<I: ImageToTensor<B>>(
        &self,
        input: &I,
        confidence_threshold: f32,
    ) -> Vec<FacialAreaRegion> {
        let nms_threshold = 0.3;
        let device = &Default::default();
        let (tensor, sizes) = resize_tensor(input.to_tensor(device), Self::DIVISOR, Self::MAX_SIZE);

        let (heatmap, scale, offset, lms) = self.model.forward(tensor);

        let (detections, lms) = self.postprocess(
            heatmap,
            lms,
            offset,
            scale,
            sizes,
            confidence_threshold,
            nms_threshold,
        );

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
    use burn::backend::NdArray;

    #[test]
    fn one_face() {
        let dataset_dir = std::env::current_dir().unwrap().join("dataset");

        let model: CenterFace<NdArray> = CenterFace::new();

        let img = image::open(dataset_dir.join("one_face.jpg")).unwrap();
        let results = model.detect(&img, 0.8);

        assert_eq!(results.len(), 1, "one face should have been detected");
    }
}
