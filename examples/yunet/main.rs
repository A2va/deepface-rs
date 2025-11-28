use burn::backend::NdArray;
use deepface::detection::{Detector, Yunet};

use image::RgbImage;

// Assumes x1 <= x2 and y1 <= y2
fn draw_rect(image: &mut RgbImage, x1: u32, x2: u32, y1: u32, y2: u32, color: &[u8; 3]) {
    for x in x1..=x2 {
        let pixel = image.get_pixel_mut(x, y1);
        *pixel = image::Rgb(*color);
        let pixel = image.get_pixel_mut(x, y2);
        *pixel = image::Rgb(*color);
    }
    for y in y1..=y2 {
        let pixel = image.get_pixel_mut(x1, y);
        *pixel = image::Rgb(*color);
        let pixel = image.get_pixel_mut(x2, y);
        *pixel = image::Rgb(*color);
    }
}

fn main() {
    let model: Yunet<NdArray> = Yunet::new();

    let img = image::open("dataset/one_face.jpg").unwrap();

    let results = model.detect(&img, 0.8, None);

    let mut img = img.to_rgb8();
    let result = results.first().unwrap();

    let x = result.x;
    let y = result.y;
    let w = result.w;
    let h = result.h;

    println!("x,y: {},{}", x, y);
    println!("w,h: {},{}", w, h);

    draw_rect(&mut img, x as u32, x + w, y, y + h, &[0, 255, 0]);

    if let Some(left_eye) = result.left_eye {
        println!("left_eye: {:?}", left_eye);
        let pixel = img.get_pixel_mut(left_eye.0, left_eye.1);
        *pixel = image::Rgb([0, 255, 0]);
    }

    if let Some(right_eye) = result.right_eye {
        println!("right_eye: {:?}", right_eye);
        let pixel = img.get_pixel_mut(right_eye.0, right_eye.1);
        *pixel = image::Rgb([0, 255, 0]);
    }

    if let Some(nose) = result.nose {
        println!("nose: {:?}", nose);
        let pixel = img.get_pixel_mut(nose.0, nose.1);
        *pixel = image::Rgb([0, 255, 0]);
    }

    if let Some(mouth_left) = result.mouth_left {
        println!("mouth_left: {:?}", mouth_left);
        let pixel = img.get_pixel_mut(mouth_left.0, mouth_left.1);
        *pixel = image::Rgb([0, 255, 0]);
    }

    if let Some(mouth_right) = result.mouth_right {
        println!("mouth_right: {:?}", mouth_right);
        let pixel = img.get_pixel_mut(mouth_right.0, mouth_right.1);
        *pixel = image::Rgb([0, 255, 0]);
    }

    img.save("output_yunet.jpg").unwrap();
}
