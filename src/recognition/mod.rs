#[cfg(feature = "deepid")]
pub mod deepid;
#[cfg(feature = "deepid")]
pub use crate::recognition::deepid::DeepID;

#[cfg(any(feature = "adaface-ir101", feature = "adaface-ir18"))]
pub mod adaface;
#[cfg(feature = "adaface-ir101")]
pub use crate::recognition::adaface::AdaFaceIR101;
#[cfg(feature = "adaface-ir18")]
pub use crate::recognition::adaface::AdaFaceIR18;

#[cfg(feature = "facenet512")]
pub mod facenet;
#[cfg(feature = "facenet512")]
pub use crate::recognition::facenet::FaceNet512;

#[cfg(feature = "dlib-recognition")]
pub mod dlib;
#[cfg(feature = "dlib-recognition")]
pub use crate::recognition::dlib::DlibRecognition;

use crate::{
    detection::{FacialAreaRegion, Landmarks},
    ImageToTensor,
};
use burn::{
    tensor::{s, Device, Int},
    Tensor,
};

#[derive(Clone, Copy, Debug)]
pub enum RecognitionModel {
    #[cfg(feature = "deepid")]
    DeepID,
    #[cfg(feature = "facenet512")]
    FaceNet512,
    #[cfg(feature = "dlib-recognition")]
    DlibRecognition,
    #[cfg(feature = "adaface-ir18")]
    AdaFaceIR18,
    #[cfg(feature = "adaface-ir101")]
    AdaFaceIR101,
}

/// Internal trait to handle recognizer metadata.
trait RecognizerMetadata {
    /// The expected input shape of the model.
    const SHAPE: (u32, u32);
}

/// A trait that all face recognition models implements
pub trait Recognizer {
    /// Generate an embedding from an input image, applying the specified normalization method if provided.
    /// The input image will be aligned using the [`FacialAreaRegion`] struct.
    fn embed<I: ImageToTensor>(
        &self,
        input: &I,
        face: FacialAreaRegion,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<1>;
}

/// Normalization methods for face embeddings
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Normalize to [0, 1]
    ZeroOne,
    /// Normalize for the FaceNet model
    FaceNet,
    /// Normalize for the FaceNet2018 model
    FaceNet2018,
    /// Normalize for the VGGFace model
    VGGFace,
    /// Normalize for the VGGFace2 model
    VGGFace2,
    /// Normalize for the ArcFace model
    ArcFace,
}

fn normalize_tensor(tensor: Tensor<4>, norm: NormalizationMethod) -> Tensor<4> {
    // Check that the tensor is between 0 and 255, rgb image
    let max = tensor.clone().max().into_scalar::<i32>();
    let min = tensor.clone().min().into_scalar::<i32>();
    assert!(max <= 255);
    assert!(min >= 0);

    match norm {
        NormalizationMethod::None => tensor,
        NormalizationMethod::ZeroOne => tensor / 255.0,
        NormalizationMethod::FaceNet => {
            let flat = tensor.clone().flatten::<2>(1, -1); // [N, C*H*W]

            // Compute mean & std per batch → shape [N,1,1,1] for broadcasting
            let mean = flat
                .clone()
                .mean_dim(1)
                .reshape([tensor.dims()[0], 1, 1, 1]);
            let std = flat.var(1).sqrt().reshape([tensor.dims()[0], 1, 1, 1]);
            (tensor - mean) / std
        }
        NormalizationMethod::FaceNet2018 => (tensor / 127.5) - 1.0,
        NormalizationMethod::VGGFace => normalize_with_means(tensor, [93.5940, 104.7624, 129.1863]),
        NormalizationMethod::VGGFace2 => {
            normalize_with_means(tensor, [91.4953, 103.8827, 131.0912])
        }
        NormalizationMethod::ArcFace => (tensor - 127.5) / 127.5,
    }
}

// Expect a tensor in the format [B, C, H, W]
fn normalize_with_means(tensor: Tensor<4>, means: [f64; 3]) -> Tensor<4> {
    // Split channels and subtract
    let mut out = tensor.clone();
    for (c, mean) in means.iter().enumerate() {
        let channel = out.clone().slice(s![.., c..c + 1, .., ..]);
        let channel: Tensor<4> = channel - *mean;
        out = out.slice_assign(s![.., c..c + 1, .., ..], channel);
    }

    out
}

pub fn crop_tensor(image: Tensor<3>, x: usize, y: usize, width: usize, height: usize) -> Tensor<3> {
    image.slice([
        0..3,          // All channels (RGB)
        y..y + height, // Height
        x..x + width,  // Width
    ])
}

/// Crops `image` (CHW) to the face region extended by a border, resizes to
/// `shape`, and aligns it via [`face_alignment`] using `face`'s sub-pixel
/// landmarks. Returns the aligned tensor in `[1, C, H, W]` form.
///
/// The border pad prevents the alignment warp (which remaps the whole output,
/// including corners outside the face) from sampling outside the source and
/// smearing border pixels as streaks.
pub(crate) fn align_face(
    image: Tensor<3>,
    face: &FacialAreaRegion,
    shape: (u32, u32),
) -> Tensor<4> {
    let Some(landmarks) = &face.landmarks else {
        panic!("face_alignment requires sub-pixel landmarks.");
    };

    let image_4d = image.unsqueeze::<4>();
    let (aligned, _) = face_alignment(image_4d, landmarks, shape);

    aligned
}

fn face_alignment(
    image: Tensor<4>,
    landmark: &Landmarks,
    image_size: (u32, u32),
) -> (Tensor<4>, Tensor<2>) {
    let device = image.device();
    let batch_size = image.dims()[0];
    let out_width = image_size.0 as usize;
    let out_height = image_size.1 as usize;

    // Get the pure-tensor transformation matrices
    let (_, inverse_transform) = estimate_norm(landmark, image_size, device);

    // Generate a sampling grid mapped to absolute pixel coordinates
    let grid = affine_grid_pixel(inverse_transform.clone(), batch_size, out_height, out_width);

    // Sample the pixels
    let aligned_face = bilinear_sample(image, grid);

    (aligned_face, inverse_transform)
}

/// The conventional ArcFace order is left first. So left first, then right eye; same for mouth.
/// The problem is that all the detectors I looked at return the right eye first,
/// so the order is inverted here in the detection models instead.
const ARCFACE_REFERENCE: Landmarks = [
    (73.5318, 51.5014), // Right eye
    (38.2946, 51.6963), // Left eye
    (56.0252, 71.7366), // Nose
    (70.7299, 92.2041), // Right mouth
    (41.5493, 92.3655), // Left mouth
];

/// Estimates both the Forward and Inverse transformation matrices.
fn estimate_norm(
    landmark: &Landmarks,
    image_size: (u32, u32),
    device: Device,
) -> (Tensor<2>, Tensor<2>) {
    let template = generate_alignment_template(image_size);

    // Convert arrays into Tensors [5, 2]. Landmarks are expected in canonical
    // order (see `Landmarks` doc), matching ARCFACE_REFERENCE.
    let src_tensor = Tensor::<2>::from_floats(landmark.map(|(x, y)| [x, y]), &device);
    let dst_tensor = Tensor::<2>::from_floats(template.map(|(x, y)| [x, y]), &device);

    // Forward Matrix: Source -> Destination
    let transform = umeyama(src_tensor.clone(), dst_tensor.clone());

    // Inverse Matrix: Destination -> Source (swap inputs = 3x3 inverse)
    let inverse_transform = umeyama(dst_tensor, src_tensor);

    (transform, inverse_transform)
}

/// Dynamically generates the target template based on the output size.
fn generate_alignment_template(image_size: (u32, u32)) -> Landmarks {
    let width = image_size.0 as f32;
    let height = image_size.1 as f32;

    let mut alignment = ARCFACE_REFERENCE;

    if width == height {
        if width % 112.0 == 0.0 {
            let ratio = width / 112.0;
            for pt in alignment.iter_mut() {
                pt.0 *= ratio;
                pt.1 *= ratio;
            }
            return alignment;
        } else if width % 128.0 == 0.0 {
            let ratio = width / 128.0;
            let diff_x = 8.0 * ratio;
            for pt in alignment.iter_mut() {
                pt.0 = (pt.0 * ratio) + diff_x;
                pt.1 *= ratio;
            }
            return alignment;
        }
    }

    // Generic fallback for all other sizes (FaceNet, DeepID, etc.) ---
    // We independently scale X and Y relative to the 112x112 base template.
    // This maps the 5 points to the exact same relative percentages of the bounding box.
    let scale_x = width / 112.0;
    let scale_y = height / 112.0;

    for pt in alignment.iter_mut() {
        pt.0 *= scale_x;
        pt.1 *= scale_y;
    }

    alignment
}

/// Generates absolute pixel coordinates for the sampling grid using pure MatMul.
fn affine_grid_pixel(
    inv_transform: Tensor<2>, // Shape: [2, 3]
    batch_size: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor<4> {
    use burn::tensor::Int;

    let device = inv_transform.device();

    // Generate output grid coordinates (X, Y)
    let x = Tensor::<1, Int>::arange(0..out_width as i64, &device).float();
    let y = Tensor::<1, Int>::arange(0..out_height as i64, &device).float();

    let x = x
        .reshape([1, out_width])
        .repeat_dim(0, out_height)
        .reshape([out_height * out_width, 1]);
    let y = y
        .reshape([out_height, 1])
        .repeat_dim(1, out_width)
        .reshape([out_height * out_width, 1]);
    let ones = Tensor::<2>::ones([out_height * out_width, 1], &device);

    // Combine into homogeneous coordinates [X, Y, 1] - Shape: [H*W, 3]
    let grid_homo = Tensor::cat(vec![x, y, ones], 1);

    // Fast mapping: X_src = X_dst * M_inv^T
    // [H*W, 3] x [3, 2] = [H*W, 2]
    let mapped_grid = grid_homo.matmul(inv_transform.transpose());

    // Reshape for the bilinear sampler
    mapped_grid
        .reshape([1, out_height, out_width, 2])
        .repeat_dim(0, batch_size)
}

/// Performs bilinear interpolation natively using Burn `gather`.
fn bilinear_sample(image: Tensor<4>, grid: Tensor<4>) -> Tensor<4> {
    let [batch_size, channels, in_h, in_w] = image.dims();
    let [_, out_h, out_w, _] = grid.dims();

    // Extract spatial coordinates from the sampling grid [B, H_out, W_out]
    let x: Tensor<3> = grid
        .clone()
        .slice([0..batch_size, 0..out_h, 0..out_w, 0..1])
        .squeeze_dim(3);
    let y: Tensor<3> = grid
        .slice([0..batch_size, 0..out_h, 0..out_w, 1..2])
        .squeeze_dim(3);

    let max_x = (in_w as f32) - 1.0;
    let max_y = (in_h as f32) - 1.0;

    // Integer neighbor coordinates, clamped to valid pixel range.
    // Floor once, derive both neighbors from the same floor.
    let x_floor = x.clone().floor();
    let y_floor = y.clone().floor();
    let x0 = x_floor.clone().clamp(0.0, max_x);
    let y0 = y_floor.clone().clamp(0.0, max_y);
    let x1 = (x_floor + 1.0).clamp(0.0, max_x);
    let y1 = (y_floor + 1.0).clamp(0.0, max_y);

    // Sub-pixel distances to each neighbor (dx0 + dx1 = 1 everywhere).
    // Weights are the product of horizontal and vertical distances.
    let dx0 = x1.clone() - x.clone();
    let dx1 = x.clone() - x0.clone();
    let dy0 = y1.clone() - y.clone();
    let dy1 = y.clone() - y0.clone();
    let w_tl = dx0.clone() * dy0.clone();
    let w_tr = dx1.clone() * dy0;
    let w_bl = dx0 * dy1.clone();
    let w_br = dx1 * dy1;

    // Flatten spatial dims for gather: [B, C, H_in * W_in]
    let img_flat: Tensor<3> = image.reshape([batch_size, channels, in_h * in_w]);

    // Flat 1D indices: idx = y * W + x
    let stride = in_w as f32;
    let idx_tl = (y0.clone() * stride + x0.clone())
        .int()
        .reshape([batch_size, out_h * out_w]);
    let idx_tr = (y0 * stride + x1.clone())
        .int()
        .reshape([batch_size, out_h * out_w]);
    let idx_bl = (y1.clone() * stride + x0)
        .int()
        .reshape([batch_size, out_h * out_w]);
    let idx_br = (y1 * stride + x1)
        .int()
        .reshape([batch_size, out_h * out_w]);

    // Expand indices to [B, C, H_out * W_out] for gather
    let idx_tl: Tensor<3, Int> = idx_tl.unsqueeze_dim(1).repeat_dim(1, channels);
    let idx_tr: Tensor<3, Int> = idx_tr.unsqueeze_dim(1).repeat_dim(1, channels);
    let idx_bl: Tensor<3, Int> = idx_bl.unsqueeze_dim(1).repeat_dim(1, channels);
    let idx_br: Tensor<3, Int> = idx_br.unsqueeze_dim(1).repeat_dim(1, channels);

    // Gather 4 nearest-neighbor pixels
    let px_tl = img_flat.clone().gather(2, idx_tl);
    let px_tr = img_flat.clone().gather(2, idx_tr);
    let px_bl = img_flat.clone().gather(2, idx_bl);
    let px_br = img_flat.gather(2, idx_br);

    // Broadcast weights to [B, C, H_out * W_out] and blend
    let w_to_3d = |w: Tensor<3>| {
        w.reshape([batch_size, 1, out_h * out_w])
            .repeat_dim(1, channels)
    };
    let out_flat = px_tl * w_to_3d(w_tl)
        + px_tr * w_to_3d(w_tr)
        + px_bl * w_to_3d(w_bl)
        + px_br * w_to_3d(w_br);

    let out_tensor = out_flat.reshape([batch_size, channels, out_h, out_w]);

    // Create a mask for invalid coordinates (out of bounds)
    let invalid_mask = x
        .clone()
        .lower_elem(0.0)
        .bool_or(x.greater_elem(max_x))
        .bool_or(y.clone().lower_elem(0.0))
        .bool_or(y.greater_elem(max_y));

    let invalid_mask_4d = invalid_mask
        .reshape([batch_size, 1, out_h, out_w])
        .repeat_dim(1, channels);

    // Fills anything outside the image with solid black
    out_tensor.mask_fill(invalid_mask_4d, 0.0)
}

pub fn umeyama(src: Tensor<2>, dst: Tensor<2>) -> Tensor<2> {
    let src_mean = src.clone().mean_dim(0); // [1, 2]
    let dst_mean = dst.clone().mean_dim(0); // [1, 2]

    let src_centered = src - src_mean.clone();
    let dst_centered = dst - dst_mean.clone();

    // Covariance matrix [2, 2]: src_centered^T * dst_centered
    let cov = src_centered.clone().transpose().matmul(dst_centered);

    // 2x2 SVD with guaranteed non-negative singular values.
    let (u, s, v_t) = analytical_svd_2x2(cov);

    // Handle reflection: if det(V*U^T) < 0 we must flip the handedness of R so
    // it stays a proper rotation (det = +1) instead of a mirror. We do this by
    // negating the second column of U; the singular values themselves stay
    // positive and are used for scale as-is.
    let vu_t = v_t
        .clone()
        .transpose()
        .matmul(u.clone().transpose())
        .unsqueeze_dim::<3>(0); // [1, 2, 2]
    let det = burn::tensor::linalg::det::<3, 2, 1>(vu_t).reshape([1]);
    let reflection_sign =
        Tensor::<1>::ones([1], &det.device()).mask_fill(det.lower_elem(0.0), -1.0);

    let u_corrected = Tensor::cat(
        vec![
            u.clone().slice([0..2, 0..1]),
            (u.slice([0..2, 1..2]) * reflection_sign.clone().reshape([1, 1])),
        ],
        1,
    ); // [2, 2]

    // Diagonal reflection matrix D = diag(1, reflection_sign)
    let device = &reflection_sign.device();
    let reflection_matrix = Tensor::cat(
        vec![
            Tensor::cat(
                vec![
                    Tensor::<2>::ones([1, 1], device),
                    Tensor::<2>::zeros([1, 1], device),
                ],
                1,
            ),
            Tensor::cat(
                vec![
                    Tensor::<2>::zeros([1, 1], device),
                    reflection_sign.clone().reshape([1, 1]),
                ],
                1,
            ),
        ],
        0,
    ); // [2, 2]

    // Rotation matrix [2, 2]: R = V * D * U^T  (canonical Umeyama form for
    // cov = src^T * dst = U * S * V^T; U * D * V^T would be its transpose).
    let rotation = v_t
        .clone()
        .transpose()
        .matmul(reflection_matrix)
        .matmul(u_corrected.transpose());

    let var_x = src_centered.powf_scalar(2.0).sum().reshape([1, 1]);

    // scale = (s0 + s1) / var_x, using the (positive) singular values
    let scale = (s.clone().slice([0..1, 0..1]) + s.slice([1..2, 0..1])) / var_x; // [1]

    // Translation [1, 2]: t = dst_mean - scale * rotation * src_mean
    let scaled_rotation = rotation * scale;
    let translation = dst_mean.transpose() - scaled_rotation.clone().matmul(src_mean.transpose());

    // Combine into [2, 3] affine matrix: [scaled_rotation | translation]
    Tensor::cat(vec![scaled_rotation, translation], 1)
}

/// Analytical 2x2 SVD: A = U * diag(s) * V^T.
/// Returns (U, s, V^T) with s guaranteed non-negative.
fn analytical_svd_2x2(cov: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
    // Returns s as a [2, 1] column vector.
    // [ a b ; c d ]
    let a = cov.clone().slice([0..1, 0..1]).reshape([1]);
    let b = cov.clone().slice([0..1, 1..2]).reshape([1]);
    let c = cov.clone().slice([1..2, 0..1]).reshape([1]);
    let d = cov.clone().slice([1..2, 1..2]).reshape([1]);

    // Eigen-angles of AᵀA (gives V) and AAᵀ (gives U).
    let ab_cd = a.clone() * b.clone() + c.clone() * d.clone();
    let theta_v = (ab_cd * 2.0).atan2(
        a.clone() * a.clone() + c.clone() * c.clone()
            - b.clone() * b.clone()
            - d.clone() * d.clone(),
    ) * 0.5;
    let ac_bd = a.clone() * c.clone() + b.clone() * d.clone();
    let theta_u = (ac_bd * 2.0).atan2(
        a.clone() * a.clone() + b.clone() * b.clone()
            - c.clone() * c.clone()
            - d.clone() * d.clone(),
    ) * 0.5;

    let (cos_v, sin_v) = (theta_v.clone().cos(), theta_v.sin());
    let (cos_u, sin_u) = (theta_u.clone().cos(), theta_u.sin());

    // V^T = [ cos_v  sin_v ; -sin_v  cos_v ]
    let v_t = Tensor::cat(
        vec![
            Tensor::cat(vec![cos_v.clone(), sin_v.clone()], 0),
            Tensor::cat(vec![-sin_v.clone(), cos_v.clone()], 0),
        ],
        0,
    )
    .reshape([2, 2]);

    // U = [ cos_u -sin_u ; sin_u  cos_u ]
    let u = Tensor::cat(
        vec![
            Tensor::cat(vec![cos_u.clone(), -sin_u.clone()], 0),
            Tensor::cat(vec![sin_u.clone(), cos_u.clone()], 0),
        ],
        0,
    )
    .reshape([2, 2]);

    // Sigma = Uᵀ A V. Its diagonal may carry signs; flip the corresponding
    // column of U so that Sigma becomes diag(+s0, +s1). This keeps
    // A = U * diag(s) * Vᵀ exact while forcing singular values positive, which
    // is what makes R = U * D * Vᵀ a proper (non-180°-off) rotation.
    let sigma = u
        .clone()
        .transpose()
        .matmul(cov.matmul(v_t.clone().transpose()));
    let sigma_00 = sigma.clone().slice([0..1, 0..1]).reshape([1]);
    let sigma_11 = sigma.slice([1..2, 1..2]).reshape([1]);
    let s = Tensor::cat(vec![sigma_00.clone().abs(), sigma_11.clone().abs()], 0).reshape([2, 1]); // [2, 1]

    let sign0 =
        Tensor::<1>::ones([1], &sigma_00.device()).mask_fill(sigma_00.lower_elem(0.0), -1.0);
    let sign1 =
        Tensor::<1>::ones([1], &sigma_11.device()).mask_fill(sigma_11.lower_elem(0.0), -1.0);

    let u = Tensor::cat(
        vec![
            u.clone().slice([0..2, 0..1]) * sign0.reshape([1, 1]),
            u.slice([0..2, 1..2]) * sign1.reshape([1, 1]),
        ],
        1,
    ); // [2, 2]

    (u, s, v_t)
}

#[cfg(test)]
mod tests {
    use super::umeyama;
    use burn::tensor::{Tensor, TensorData};

    // A clean similarity: dst = scale * R * src + t, with R a rotation and
    // scale a uniform factor. `umeyama(src, dst)` must recover it exactly
    // (map every src point onto its dst point). This catches the classic
    // Umeyama transpose bug (R built as U*D*V^T instead of V*D*U^T), which
    // leaves landmarks misaligned by a rotation/reflection.
    #[test]
    fn umeyama_recovers_similarity() {
        let src = [
            [100.0, 80.0],
            [40.0, 90.0],
            [60.0, 120.0],
            [90.0, 140.0],
            [50.0, 130.0],
        ];
        // rotate 15deg, scale 1.2, translate
        let (c, s) = (15.0_f32.to_radians().cos(), 15.0_f32.to_radians().sin());
        let scale = 1.2_f32;
        let (tx, ty) = (30.0_f32, 10.0_f32);
        let mut dst = [[0.0_f32; 2]; 5];
        for (i, [x, y]) in src.iter().enumerate() {
            dst[i][0] = (scale * (c * x - s * y) + tx) as f32;
            dst[i][1] = (scale * (s * x + c * y) + ty) as f32;
        }

        let device = Default::default();
        let src_flat: Vec<f32> = src.iter().flat_map(|p| p.iter().copied()).collect();
        let src_t = Tensor::<2>::from_data(TensorData::new(src_flat.clone(), [5, 2]), &device);
        let dst_flat: Vec<f32> = dst.iter().flat_map(|p| p.iter().copied()).collect();
        let dst_t = Tensor::<2>::from_data(TensorData::new(dst_flat.clone(), [5, 2]), &device);
        let m = umeyama(src_t.clone(), dst_t.clone()); // [2, 3] affine src -> dst

        // Apply the same way affine_grid_pixel does: dst_row = src_row * R^T + t.
        let mapped = src_t
            .matmul(m.clone().slice([0..2, 0..2]).transpose())
            .add(m.slice([0..2, 2..3]).transpose());
        let max_err = mapped.sub(dst_t).abs().max().into_scalar::<f32>();
        assert!(
            max_err < 1e-2,
            "umeyama failed to recover the similarity transform, max err {}",
            max_err
        );
    }
}
