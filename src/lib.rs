//! A rust implementation of the [deepface](https://github.com/serengil/deepface) python library.
//! # Supported models
//! Detection:
//! * CenterFace
//! * Dlib
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

use burn::prelude::{Backend, Device};
use burn::tensor::{Element, Tensor, TensorData};
use image::DynamicImage;

/// Trait to convert an image-like input into a 3D tensor with shape `[C, H, W]`
///
/// This trait is implemented for `image::DynamicImage` and `burn::tensor::Tensor`,
/// allowing consistent conversion across image inputs and tensor data.
///
/// The resulting tensor is `[C, H, W]`.
pub trait ImageToTensor<B: Backend> {
    /// The ouput tensor format is [C, H, W], where C is the number of channel,
    /// H the height and W the width
    fn to_tensor(&self, device: &<B as Backend>::Device) -> Tensor<B, 3>;
}

/// Converts a `DynamicImage` to a tensor of shape `[C, H, W]`, in RGB format.
impl<B: Backend> ImageToTensor<B> for DynamicImage {
    fn to_tensor(&self, device: &<B as Backend>::Device) -> Tensor<B, 3> {
        let rgb_image = self.to_rgb8();

        // Convert image data to tensor
        let data = rgb_image.into_raw();

        // Create tensor from image data [H, W, C] and reshape to [C, H, W]
        let tensor = to_tensor(
            data,
            [self.height() as usize, self.width() as usize, 3],
            device,
        );
        tensor
    }
}

/// Clones the tensor to the specified device. Assumes input is already `[C, H, W]`.
impl<B: Backend> ImageToTensor<B> for Tensor<B, 3> {
    // The tensor must be in 3 dimensions [C, H, W]
    fn to_tensor(&self, device: &<B as Backend>::Device) -> Tensor<B, 3> {
        self.clone().to_device(device)
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

use burn::vision::utils::ImageDimOrder;

pub enum ImageTensor<B: Backend> {
    /// dims: (height, width)
    Hw(Tensor<B, 2>),
    /// dims: (channels, height, width)
    Chw(Tensor<B, 3>),
    /// dims: (height, width, channels)
    Hwc(Tensor<B, 3>),
    /// dims: (batch_size, height, width)
    Nhw(Tensor<B, 3>),
    /// dims: (batch_size, channels, height, width)
    Nchw(Tensor<B, 4>),
    /// dims: (batch_size, height, width, channels)
    Nhwc(Tensor<B, 4>),
}

impl<B: Backend> ImageTensor<B> {
    pub fn to_dim_order(self, dim_order: ImageDimOrder) -> ImageTensor<B> {
        match self {
            ImageTensor::Hw(tensor) => match dim_order {
                ImageDimOrder::Hw => ImageTensor::Hw(tensor),
                ImageDimOrder::Chw => ImageTensor::Chw(tensor.unsqueeze::<3>()),
                ImageDimOrder::Hwc => ImageTensor::Hwc(tensor.unsqueeze_dim(1)),
                ImageDimOrder::Nhw => ImageTensor::Nhw(tensor.unsqueeze::<3>()),
                ImageDimOrder::Nchw => ImageTensor::Nchw(tensor.unsqueeze::<4>()),
                ImageDimOrder::Nhwc => ImageTensor::Nhwc(tensor.unsqueeze_dims(&[0, -1])),
            },
            _ => todo!("Implement other ImageTensor input variants"),
        }
    }
}

impl<B: Backend> From<ImageTensor<B>> for ImageDimOrder {
    fn from(image_tensor: ImageTensor<B>) -> Self {
        match image_tensor {
            ImageTensor::Hw(_) => ImageDimOrder::Hw,
            ImageTensor::Chw(_) => ImageDimOrder::Chw,
            ImageTensor::Hwc(_) => ImageDimOrder::Hwc,
            ImageTensor::Nhw(_) => ImageDimOrder::Nhw,
            ImageTensor::Nchw(_) => ImageDimOrder::Nchw,
            ImageTensor::Nhwc(_) => ImageDimOrder::Nhwc,
        }
    }
}
