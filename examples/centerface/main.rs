use deepface_rs::detection::{CenterFace, Detector};
use image::ImageBuffer;

// Assumes x1 <= x2 and y1 <= y2
fn draw_rect(
    image: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x1: u32,
    x2: u32,
    y1: u32,
    y2: u32,
    color: &[u8; 3],
) {
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
    let model = CenterFace::new();

    let img = image::open("dataset/img1.jpg").unwrap();

    let results  = model.detect(&img);
    

    let mut img = img.to_rgb8();
    let result = results.first().unwrap();

    let x = result.x;
    let y = result.y;
    let w = result.w;
    let h = result.h;

    println!("x,y: {},{}", x, y);
    println!("w,h: {},{}", w, h);

    println!("left_eye: {:?}", result.left_eye.unwrap_or((0, 0)));

    draw_rect(&mut img, x as u32, x + w, y, y + h, &[0, 255, 0]);
    img.save("output_centerface.jpg").unwrap();
}
