pub mod detection;
pub mod recognition;

use burn::prelude::{Backend, Device};
use burn::tensor::{Element, Tensor, TensorData};
use image::DynamicImage;

/// Trait to convert an image-like input into a 3D tensor with shape `[C, H, W]`
/// (channel-first format), suitable for deep learning models.
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
