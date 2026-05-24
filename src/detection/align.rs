use burn::tensor::backend::Backend;
use burn::tensor::grid::affine_grid_2d;
use burn::tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn::tensor::Tensor;
use burn::vision::utils::TensorDisplayOptions;
use burn::vision::Transform2D;

use super::FacialAreaRegion;

// #[derive(Clone, Debug)]
// pub struct DetectedFace<B: Backend> {
//     pub img: Tensor<B, 3>,
//     pub facial_area: FacialAreaRegion,
//     pub confidence: Option<f32>,
// }

pub fn extract_face<B: Backend>(
    facial_area: FacialAreaRegion,
    img: &Tensor<B, 3>,
    align: bool,
    expand_percentage: u32,
    width_border: u32,
    height_border: u32,
) -> FacialAreaRegion {
    let mut x = facial_area.x;
    let mut y = facial_area.y;
    let mut w = facial_area.w;
    let mut h = facial_area.h;

    let mut left_eye = facial_area.left_eye;
    let mut right_eye = facial_area.right_eye;
    let mut nose = facial_area.nose;
    let mut mouth_left = facial_area.mouth_left;
    let mut mouth_right = facial_area.mouth_right;
    let confidence = facial_area.confidence;

    let [c, img_h, img_w] = img.dims();

    if expand_percentage > 0 {
        let expanded_w = w + (w * expand_percentage) / 100;
        let expanded_h = h + (h * expand_percentage) / 100;

        let diff_w = (expanded_w - w) / 2;
        let diff_h = (expanded_h - h) / 2;

        x = x.saturating_sub(diff_w);
        y = y.saturating_sub(diff_h);
        w = ((img_w as u32).saturating_sub(x)).min(expanded_w);
        h = ((img_h as u32).saturating_sub(y)).min(expanded_h);
    }

    // Convert to usize just for the tensor slice operation
    let mut detected_face = img.clone().slice([
        0..c,
        (y as usize)..(y + h) as usize,
        (x as usize)..(x + w) as usize,
    ]);

    if align {
        let (sub_img, relative_x, relative_y) = extract_sub_image(img, (x, y, w, h));

        let (aligned_sub_img, angle) = align_img_wrt_eyes(&sub_img, left_eye, right_eye);

        let sub_dims = aligned_sub_img.dims();
        let (rotated_x1, rotated_y1, rotated_x2, rotated_y2) = project_facial_area(
            (relative_x, relative_y, relative_x + w, relative_y + h),
            angle,
            (sub_dims[1] as u32, sub_dims[2] as u32),
        );

        detected_face = aligned_sub_img.slice([
            0..c,
            rotated_y1 as usize..rotated_y2 as usize,
            rotated_x1 as usize..rotated_x2 as usize,
        ]);

        // Restore coords before border was added safely
        x = x.saturating_sub(width_border);
        y = y.saturating_sub(height_border);

        let adjust_pt = |pt: Option<(u32, u32)>| -> Option<(u32, u32)> {
            pt.map(|(px, py)| {
                (
                    px.saturating_sub(width_border),
                    py.saturating_sub(height_border),
                )
            })
        };

        left_eye = adjust_pt(left_eye);
        right_eye = adjust_pt(right_eye);
        nose = adjust_pt(nose);
        mouth_left = adjust_pt(mouth_left);
        mouth_right = adjust_pt(mouth_right);
    }

    // Save the tensor as image
    let [_, hh, ww] = detected_face.dims();
    use burn::vision::utils::{
        save_tensor_as_image, ColorDisplayOpts, ImageDimOrder, TensorDisplayOptions,
    };
    save_tensor_as_image(
        detected_face,
        TensorDisplayOptions {
            dim_order: ImageDimOrder::Chw,
            width_out: ww,
            height_out: hh,
            color_opts: ColorDisplayOpts::Rgb,
            batch_opts: None,
        },
        "aligned_face.jpg",
    )
    .unwrap();

    // DetectedFace {
    //     img: detected_face,
    //     facial_area: FacialAreaRegion {
    //         x,
    //         y,
    //         w,
    //         h,
    //         confidence,
    //         left_eye,
    //         right_eye,
    //         nose,
    //         mouth_left,
    //         mouth_right,
    //     },
    //     confidence,
    // }
    //
    FacialAreaRegion {
        x,
        y,
        w,
        h,
        confidence,
        left_eye,
        right_eye,
        nose,
        mouth_left,
        mouth_right,
    }
}

pub fn extract_sub_image<B: Backend>(
    img: &Tensor<B, 3>,
    facial_area: (u32, u32, u32, u32),
) -> (Tensor<B, 3>, u32, u32) {
    let [c, h_img, w_img] = img.dims();
    let (x, y, w, h) = facial_area;

    let relative_x = w / 2;
    let relative_y = h / 2;

    // Use i64 for safe math since x1 or y1 could be negative
    let x1 = x as i64 - relative_x as i64;
    let y1 = y as i64 - relative_y as i64;
    let x2 = x as i64 + w as i64 + relative_x as i64;
    let y2 = y as i64 + h as i64 + relative_y as i64;

    if x1 >= 0 && y1 >= 0 && (x2 as usize) <= w_img && (y2 as usize) <= h_img {
        let sub_img = img
            .clone()
            .slice([0..c, y1 as usize..y2 as usize, x1 as usize..x2 as usize]);
        return (sub_img, relative_x, relative_y);
    }

    let safe_x1 = x1.max(0) as usize;
    let safe_y1 = y1.max(0) as usize;
    let safe_x2 = (x2 as usize).min(w_img);
    let safe_y2 = (y2 as usize).min(h_img);

    let cropped_region = img
        .clone()
        .slice([0..c, safe_y1..safe_y2, safe_x1..safe_x2]);

    let out_h = (h + 2 * relative_y) as usize;
    let out_w = (w + 2 * relative_x) as usize;

    let device = img.device();
    let mut extracted_face = Tensor::<B, 3>::zeros([c, out_h, out_w], &device);

    let start_x = if x1 < 0 {
        (relative_x as i64 - x as i64).max(0) as usize
    } else {
        0
    };
    let start_y = if y1 < 0 {
        (relative_y as i64 - y as i64).max(0) as usize
    } else {
        0
    };

    let crop_h = safe_y2 - safe_y1;
    let crop_w = safe_x2 - safe_x1;

    extracted_face = extracted_face.slice_assign(
        [0..c, start_y..start_y + crop_h, start_x..start_x + crop_w],
        cropped_region,
    );

    (extracted_face, relative_x, relative_y)
}

pub fn align_img_wrt_eyes<B: Backend>(
    img: &Tensor<B, 3>,
    left_eye: Option<(u32, u32)>,
    right_eye: Option<(u32, u32)>,
) -> (Tensor<B, 3>, f32) {
    let [c, h, w] = img.dims();
    if h == 0 || w == 0 {
        return (img.clone(), 0.0);
    }

    // DEBUG: Check if landmarks are actually being provided by your detector!
    println!(
        "DEBUG - Alignment Eyes -> Left: {:?}, Right: {:?}",
        left_eye, right_eye
    );

    if let (Some(le), Some(re)) = (left_eye, right_eye) {
        let dy = le.1 as f32 - re.1 as f32;
        let dx = le.0 as f32 - re.0 as f32;

        // Prevent NaN if the points are identical
        if dx == 0.0 && dy == 0.0 {
            return (img.clone(), 0.0);
        }

        let angle_rad = dy.atan2(dx);
        let angle_deg = angle_rad.to_degrees();
        println!("DEBUG - Computed Rotation Angle: {} degrees", angle_deg);

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let h_f32 = h as f32;
        let w_f32 = w as f32;

        // Burn's affine grid maps Target -> Source in normalized [-1, 1] space.
        // We MUST multiply the sine terms by the aspect ratio to prevent the image
        // from skewing/stretching if the crop is a rectangle (which it usually is).
        let theta_data: [[[f32; 3]; 2]; 1] = [[
            [cos_a, -sin_a * (h_f32 / w_f32), 0.0],
            [sin_a * (w_f32 / h_f32), cos_a, 0.0],
        ]];

        let device = img.device();
        // DeepFace uses OpenCV which rotates the image Counter-Clockwise.
        // Burn's grid_sample maps the Target grid back to the Source grid.
        // To visually rotate an image Counter-Clockwise, we must rotate the grid Clockwise.
        // Transform2D::rotation applies a CCW rotation, so we negate the angle.
        let theta = -angle_rad;

        // Correct for Aspect Ratio so non-square crops do not stretch or squish.
        let aspect_ratio = w as f32 / h as f32;

        let t_fwd = Transform2D::scale(aspect_ratio, 1.0, 0.0, 0.0);
        let t_rot = Transform2D::rotation(theta, 0.0, 0.0);
        let t_inv = Transform2D::scale(1.0 / aspect_ratio, 1.0, 0.0, 0.0);

        // `composed` applies right-to-left: t_inv * t_rot * t_fwd
        let transform = Transform2D::composed([t_inv, t_rot, t_fwd]);

        // Transform2D requires a 4D tensor [Batch, Channels, Height, Width]
        let img_4d = img.clone().reshape([1, c, h, w]);
        let aligned_4d = transform.transform(img_4d);
        let aligned_3d = aligned_4d.reshape([c, h, w]);

        (aligned_3d, angle_deg)
    } else {
        println!("DEBUG - Left or Right eye was None, skipping rotation.");
        (img.clone(), 0.0)
    }
}

pub fn project_facial_area(
    facial_area: (u32, u32, u32, u32), // x1, y1, x2, y2
    angle: f32,
    size: (u32, u32), // height, width
) -> (u32, u32, u32, u32) {
    let direction = if angle >= 0.0 { 1.0 } else { -1.0 };
    let abs_angle = angle.abs() % 360.0;
    if abs_angle == 0.0 {
        return facial_area;
    }

    let angle_rad = abs_angle * std::f32::consts::PI / 180.0;
    let (height, width) = size;
    let height_f32 = height as f32;
    let width_f32 = width as f32;

    let x = (facial_area.0 as f32 + facial_area.2 as f32) / 2.0 - width_f32 / 2.0;
    let y = (facial_area.1 as f32 + facial_area.3 as f32) / 2.0 - height_f32 / 2.0;

    let x_new = x * angle_rad.cos() + y * direction * angle_rad.sin();
    let y_new = -x * direction * angle_rad.sin() + y * angle_rad.cos();

    let x_new = x_new + width_f32 / 2.0;
    let y_new = y_new + height_f32 / 2.0;

    let half_w = (facial_area.2 as f32 - facial_area.0 as f32) / 2.0;
    let half_h = (facial_area.3 as f32 - facial_area.1 as f32) / 2.0;

    let x1 = x_new - half_w;
    let y1 = y_new - half_h;
    let x2 = x_new + half_w;
    let y2 = y_new + half_h;

    let x1_clamped = x1.max(0.0) as u32;
    let y1_clamped = y1.max(0.0) as u32;
    let x2_clamped = x2.max(0.0) as u32;
    let y2_clamped = y2.max(0.0) as u32;

    let x2_final = x2_clamped.min(width);
    let y2_final = y2_clamped.min(height);

    (x1_clamped, y1_clamped, x2_final, y2_final)
}
