//! A rust implementation of the [deepface](https://github.com/serengil/deepface) python library.
//! # Supported models
//! Detection:
//! * CenterFace
//! * Dlib (`dlib-detection`)
//! * Yunet
//!
//! Recognition:
//! * DeepID
//! * FaceNet512
//!
//! To use one of these model, you must add them as features in your `Cargo.toml`:
//! ```toml
//! deepface = {git = "https://github.com/A2va/deepface-rs", features = ["yunet", "facenet512"]}
//! ```

pub mod detection;
pub mod recognition;

pub mod metrics;

use burn::tensor::{Device, Element, Tensor, TensorData};
use image::{DynamicImage, RgbImage, SubImage};

/// Trait to convert an image-like input into a 3D tensor with shape `[C, H, W]`
///
/// This trait is implemented for `image::DynamicImage` and `burn::tensor::Tensor`,
/// allowing consistent conversion across image inputs and tensor data.
///
/// The resulting tensor is `[C, H, W]`.
pub trait ImageToTensor {
    /// The ouput tensor format is [C, H, W], where C is the number of channel,
    /// H the height and W the width
    fn to_tensor(&self) -> Tensor<3>;
}

/// Converts a `DynamicImage` to a tensor of shape `[C, H, W]`, in RGB format.
impl ImageToTensor for DynamicImage {
    fn to_tensor(&self) -> Tensor<3> {
        let rgb_image = self.to_rgb8();
        rgb_image.to_tensor()
    }
}

impl ImageToTensor for RgbImage {
    fn to_tensor(&self) -> Tensor<3> {
        // Convert image data to tensor
        let data = self.clone().into_raw();
        let device = &Default::default();
        // Create tensor from image data [H, W, C] and reshape to [C, H, W]
        let tensor = to_tensor(
            data,
            [self.height() as usize, self.width() as usize, 3],
            device,
        );
        tensor
    }
}

impl ImageToTensor for SubImage<&mut DynamicImage> {
    fn to_tensor(&self) -> Tensor<3> {
        let img = DynamicImage::ImageRgba8(self.to_image());
        img.to_tensor()
    }
}

impl ImageToTensor for SubImage<&DynamicImage> {
    fn to_tensor(&self) -> Tensor<3> {
        let img = DynamicImage::ImageRgba8(self.to_image());
        img.to_tensor()
    }
}

/// Clones the tensor to the specified device. Assumes input is already `[C, H, W]`.
impl ImageToTensor for Tensor<3> {
    // The tensor must be in 3 dimensions [C, H, W]
    fn to_tensor(&self) -> Tensor<3> {
        self.clone()
    }
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
pub fn to_tensor<T: Element>(data: Vec<T>, shape: [usize; 3], device: &Device) -> Tensor<3> {
    Tensor::<3>::from_data(TensorData::new(data, shape).convert::<f32>(), device)
        // [H, W, C] -> [C, H, W]
        .permute([2, 0, 1])
}

// This was enum was moved here because both the detection and recognition model of dlib needs it.
#[cfg(any(feature = "dlib-detection", feature = "dlib-recognition"))]
/// Dlib detector model type.
pub enum DlibDetectorModel {
    /// Based on the Convolutional Neural Network (CNN).
    Cnn,
    /// Based on the Histogram of Oriented Gradients (HOG).
    Hog,
}
