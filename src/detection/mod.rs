#[cfg(feature = "centerface")]
pub mod centerface;
#[cfg(feature = "centerface")]
pub use crate::detection::centerface::CenterFace;

#[cfg(feature = "yunet")]
pub mod yunet;
#[cfg(feature = "yunet")]
pub use crate::detection::yunet::Yunet;

#[cfg(any(feature = "scrfd-10g", feature = "scrfd-500m"))]
pub mod scrfd;
#[cfg(feature = "scrfd-10g")]
pub use crate::detection::scrfd::Scrfd10g;
#[cfg(feature = "scrfd-500m")]
pub use crate::detection::scrfd::Scrfd500m;

#[cfg(feature = "dlib-detection")]
pub mod dlib;
#[cfg(feature = "dlib-detection")]
pub use crate::detection::dlib::DlibDetection;

pub use burn::vision::NmsOptions;

use crate::ImageToTensor;

/// Internal trait to handle detector metadata.
trait DetectorMetadata {
    /// The size‐rounding multiple (e.g. 32)
    const DIVISOR: u32;
    /// Optional max side (e.g. 640 for Yunet)
    const MAX_SIZE: Option<u32>;
}

/// A trait that all face dectector models implements
pub trait Detector {
    /// Detect faces in an input image, returning bounding boxes and landmarks.
    /// - `input`: The input image implementing `ImageToTensor`, tensor are also accepted
    /// If you want your tensor to be on a specific device, you must set the device for that tensor before calling this function.
    /// It is not possible to choose the device for an image, it will use the default one for that backend.
    /// - `confidence_threshold`: Minimum confidence to consider a detection valid
    /// - `nms_threshold`: Optional IoU threshold for non-maximum suppression
    fn detect<I: ImageToTensor>(&self, input: &I, nms_options: NmsOptions)
        -> Vec<FacialAreaRegion>;
}

#[derive(Clone, Debug)]
pub struct FacialAreaRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub confidence: Option<f32>,
    /// Sub-pixel facial landmarks in canonical order. Query semantic regions
    /// (eye centers, outlines, ...) via [`Landmarks::part`]/[`Landmarks::center`]
    /// and the ArcFace points via [`Landmarks::to_5_points`].
    pub landmarks: Option<Landmarks>,
}

impl FacialAreaRegion {
    fn center(&self, part: crate::landmarks::FacePart) -> Option<(u32, u32)> {
        self.landmarks
            .as_ref()
            .and_then(|lm| lm.center(part))
            .map(|(x, y)| (x as u32, y as u32))
    }

    /// Person's right eye center.
    pub fn right_eye(&self) -> Option<(u32, u32)> {
        self.center(crate::landmarks::FacePart::RightEye)
    }

    /// Person's left eye center.
    pub fn left_eye(&self) -> Option<(u32, u32)> {
        self.center(crate::landmarks::FacePart::LeftEye)
    }

    /// Nose tip.
    pub fn nose(&self) -> Option<(u32, u32)> {
        self.center(crate::landmarks::FacePart::NoseTip)
    }

    /// Person's right mouth corner.
    pub fn right_mouth(&self) -> Option<(u32, u32)> {
        self.center(crate::landmarks::FacePart::RightMouth)
    }

    /// Person's left mouth corner.
    pub fn left_mouth(&self) -> Option<(u32, u32)> {
        self.center(crate::landmarks::FacePart::LeftMouth)
    }
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

use burn::Tensor;

/// Resize a tensor to match model input requirements.
/// The tensor shape is expected to be [C, H, W] and will be resized to [1, C, new_H, new_W].
fn resize_tensor(
    tensor: Tensor<3>,
    divisor: u32,
    max_size: Option<u32>,
) -> (Tensor<4>, ResizedDimensions) {
    use burn::nn::interpolate::Interpolate2dConfig;
    let (width, height) = (tensor.dims()[2] as u32, tensor.dims()[1] as u32);
    let sizes = resize_to_divisor_multiple(width, height, divisor, max_size);

    let interpolate = Interpolate2dConfig::new()
        .with_output_size(Some([sizes.height as usize, sizes.width as usize]))
        .init();

    (interpolate.forward(tensor.unsqueeze::<4>()), sizes) // [B, C, H, W]
}

/// Canonical (5-point) facial landmarks. See [`crate::landmarks::Landmarks`].
pub use crate::landmarks::Landmarks;

struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
}
