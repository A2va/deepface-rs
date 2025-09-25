use burn::backend::NdArray;
use burn::tensor::linalg::cosine_similarity;
use deepface::recognition::{FaceNet512, Recognizer};

fn main() {
    let model: FaceNet512<NdArray> = FaceNet512::new();
    let img = image::open("dataset/cun.png").unwrap();

    let result1 = model.embed(&img, None);

    let img = image::open("dataset/image.png").unwrap();
    let result2 = model.embed(&img, None);

    let simularity = cosine_similarity(result1, result2, -1, None);
    println!("Distance: {}", simularity);
}