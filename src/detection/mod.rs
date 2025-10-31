#[cfg(feature = "centerface")]
pub mod centerface;
#[cfg(feature = "centerface")]
pub use crate::detection::centerface::CenterFace;

#[cfg(feature = "yunet")]
pub mod yunet;
#[cfg(feature = "yunet")]
pub use crate::detection::yunet::Yunet;

#[cfg(feature = "dlib")]
pub mod dlib;
#[cfg(feature = "dlib")]
pub use crate::detection::dlib::Dlib;

use crate::ImageToTensor;
use image::{DynamicImage, ImageBuffer, Rgb, SubImage};

/// A trait that all face dectector models implements
pub trait Detector<B: Backend> {
    /// The size‐rounding multiple (e.g. 32)
    const DIVISOR: u32;
    /// Optional max side (e.g. 640 for Yunet)
    const MAX_SIZE: Option<u32>;
 
    /// Detect faces in an input image, returning bounding boxes and landmarks.
    /// - `input`: The input image implementing `ImageToTensor`, tensor are also accepted
    /// - `confidence_threshold`: Minimum confidence to consider a detection valid
    /// - `nms_threshold`: Optional IoU threshold for non-maximum suppression
    fn detect<I: ImageToTensor<B>>(
        &self,
        input: &I,
        confidence_threshold: f32,
        nms_threshold: Option<f32>,
    ) -> Vec<FacialAreaRegion>;
}

pub struct FacialAreaRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub left_eye: Option<(u32, u32)>,
    pub right_eye: Option<(u32, u32)>,
    pub confidence: Option<f32>,
    pub nose: Option<(u32, u32)>,
    pub mouth_right: Option<(u32, u32)>,
    pub mouth_left: Option<(u32, u32)>,
}

pub struct DetectedFace {
    // TODO Explore SubImage for the DetectedFace, also what is confidence if it's present in FacialAreaRegion
    img: DynamicImage,
    facial_area: FacialAreaRegion,
    confidence: f32,
}

/// Represents resized dimensions and scale factors.
#[derive(Debug, Clone, Copy)]
struct ResizedDimensions {
    height: u32,
    width: u32,
    height_scale: f32,
    width_scale: f32,
}

/// Resizes dimensions to multiples of a divisor while preserving aspect ratio.
///
/// # Arguments
///
/// * `width` - Original width
/// * `height` - Original height
/// * `divisor` - Value that new dimensions should be multiples of (typically 32)
/// * `max_dimension` - Optional maximum size for either dimension
///
/// # Returns
///
/// (new_height, new_width, height_scale, width_scale)
fn resize_to_divisor_multiple(
    original_width: u32,
    original_height: u32,
    divisor: u32,
    max_dimension: Option<u32>,
) -> ResizedDimensions {
    let mut scaled_width = original_width as f32;
    let mut scaled_height = original_height as f32;

    // Apply max dimension constraint
    if let Some(max) = max_dimension {
        if original_height > max || original_width > max {
            let scale_ratio = max as f32 / scaled_height.max(scaled_width);
            scaled_width *= scale_ratio;
            scaled_height *= scale_ratio;
        }
    }

    let adjusted_height = f32::ceil(scaled_height / divisor as f32) * divisor as f32;
    let adjusted_width = f32::ceil(scaled_width / divisor as f32) * divisor as f32;

    let width_scale = adjusted_width / original_width as f32;
    let height_scale = adjusted_height / original_height as f32;

    ResizedDimensions {
        width: adjusted_width as u32,
        height: adjusted_height as u32,
        height_scale: height_scale,
        width_scale: width_scale,
    }
}

use burn::prelude::{Backend, Tensor};

/// Resize a tensor to match model input requirements.
/// The tensor shape is expected to be [C, H, W] and will be resized to [1, C, new_H, new_W].
fn resize_tensor<B: Backend>(
    tensor: Tensor<B, 3>,
    divisor: u32,
    max_size: Option<u32>,
) -> (Tensor<B, 4>, ResizedDimensions) {
    use burn::nn::interpolate::Interpolate2dConfig;
    let (width, height) = (tensor.dims()[2] as u32, tensor.dims()[1] as u32);
    let sizes = resize_to_divisor_multiple(width, height, divisor, max_size);

    let interpolate = Interpolate2dConfig::new()
        .with_output_size(Some([sizes.height as usize, sizes.width as usize]))
        .init();

    (interpolate.forward(tensor.unsqueeze::<4>()), sizes) // [B, C, H, W]
}

/// Represents a set of facial landmarks as 2D coordinates.
///
/// The landmarks are typically ordered as:
/// - Right eye
/// - Left eye
/// - Nose
/// - Right mouth corner
/// - Left mouth corner
///
/// Note that the order is not strictly enforced, and the array contains 5 points
/// represented as floating-point (x, y) coordinates.
type Landmarks = [(f32, f32); 5];

struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
}

/// Intersection over union of two bounding boxes.
fn iou(b1: &BoundingBox, b2: &BoundingBox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

/// Perform non-maximum suppression over boxes of the same class.
fn non_maximum_suppression(
    bboxes: &mut Vec<BoundingBox>,
    lms: &mut Vec<Landmarks>,
    threshold: f32,
) {
    bboxes.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
    let mut current_index = 0;
    for index in 0..bboxes.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let iou = iou(&bboxes[prev_index], &bboxes[index]);
            if iou > threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            bboxes.swap(current_index, index);
            lms.swap(current_index, index);
            current_index += 1;
        }
    }
    bboxes.truncate(current_index);
    lms.truncate(current_index);
}

fn add_border(img: &DynamicImage) -> DynamicImage {
    let img = img.to_rgb8();
    let width = img.width();
    let height = img.height();

    let width_border = (0.5 * width as f32) as u32;
    let height_border = (0.5 * height as f32) as u32;

    // Create a new image with borders
    let new_width = width + 2 * width_border;
    let new_height = height + 2 * height_border;

    let mut new_img = ImageBuffer::new(new_width, new_height);

    // Fill the entire image with black (border color)
    for pixel in new_img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]); // Black color for border
    }

    // Copy the original image to the center of the new image
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            new_img.put_pixel(x + width_border, y + height_border, *pixel);
        }
    }
    new_img.into()
}
