use burn::backend::NdArray;
use deepface::recognition::{DeepID, Recognizer};
use burn::tensor::linalg::cosine_similarity;

fn main() {
    let model: DeepID<NdArray> = DeepID::new();
    let img = image::open("dataset/cun.png").unwrap();

    let result1 = model.embed(&img, None);

    let img = image::open("dataset/cun2.jpg").unwrap();
    let result2 = model.embed(&img, None);

    let simularity = cosine_similarity(result1, result2, -1, None);
    println!("Distance: {}", simularity);
}