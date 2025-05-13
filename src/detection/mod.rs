pub mod centerface;
pub use crate::detection::centerface::CenterFace;

pub mod yunet;
pub use crate::detection::yunet::Yunet;

pub trait Detector {
    fn detect(&self, input: &DynamicImage) -> Vec<FacialAreaRegion>;
}

use image::{DynamicImage, ImageBuffer, Rgb, SubImage};


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
    i: SubImage<DynamicImage>,
    img: DynamicImage,
    facial_area: FacialAreaRegion,
    confidence: f32,
}

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Device, Element, Tensor, TensorData},
};

type DeepFaceBackend = NdArray;

/// Resizes dimensions to be multiples of a specified divisor.
///
/// This function takes input width and height dimensions and calculates new dimensions
/// that are multiples of the specified divisor (typically 32 for many ML models).
/// It also returns the scaling factors needed to convert between original and new dimensions.
///
/// # Arguments
///
/// * `width` - The original width
/// * `height` - The original height
/// * `divisor` - The value that the new dimensions should be multiples of
///
/// # Returns
///
/// A tuple containing:
/// * `new_height` - The adjusted height (multiple of divisor)
/// * `new_width` - The adjusted width (multiple of divisor)
/// * `scale_height` - The scaling factor for height (new_height / original_height)
/// * `scale_width` - The scaling factor for width (new_width / original_width)
///
/// Note: Height is returned before width to match the convention used in many ML models.
fn resize_to_multiple_of_divisor(
    width: u32,
    height: u32,
    divisor: u32,
    max_size: Option<u32>,
) -> (u32, u32, f32, f32) {
    let mut new_width = width as f32;
    let mut new_height = height as f32;

    let new_height = f32::ceil(height / divisor as f32) * divisor as f32;
    let new_width = f32::ceil(width / divisor as f32) * divisor as f32;

    let scale_width = new_width / width as f32;
    let scale_height = new_height / height as f32;
    // Return the height first, because in ml models,
    // the height is often in front of the width.
    (
        new_height as u32,
        new_width as u32,
        scale_height,
        scale_width,
    )
}

/// Converts a vector of data into a 3D tensor with optional permutation.
///
/// # Arguments
///
/// * `data` - A vector of elements to be converted into a tensor
/// * `shape` - The original shape of the tensor as [height, width, channels]
/// * `device` - The backend device where the tensor will be created
///
/// # Returns
///
/// A 3D tensor with data converted to the backend's float element type and permuted from [H, W, C] to [C, H, W]
///
/// # Note
///
/// This function is useful for preparing image data for machine learning models by converting
/// raw data to a tensor and rearranging the dimensions to match model input requirements.
fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(data, shape).convert::<B::FloatElem>(),
        device,
    )
    // [H, W, C] -> [C, H, W]
    .permute([2, 0, 1])
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
