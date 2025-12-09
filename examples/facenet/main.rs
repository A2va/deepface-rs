use burn::{backend::NdArray, Tensor};
use image::{DynamicImage, GenericImage};

use deepface::detection::{Detector, Yunet};
use deepface::metrics::{verify, DistanceMethod};
use deepface::recognition::{FaceNet512, RecognitionModel, Recognizer};

fn embed(mut img: DynamicImage) -> Tensor<NdArray, 1> {
    let model: Yunet<NdArray> = Yunet::new();
    let results = model.detect(&img, 0.8, None);
    let results = results.first().unwrap();

    let subimg = img.sub_image(results.x, results.y, results.w, results.h);

    let model: FaceNet512<NdArray> = FaceNet512::new();
    model.embed(&subimg, None)
}

fn main() {
    let img = image::open("dataset/img1.jpg").unwrap();
    let result1 = embed(img);

    let img = image::open("dataset/img2.jpg").unwrap();
    let result2 = embed(img);

    let d = verify(
        result1,
        result2,
        RecognitionModel::FaceNet512,
        DistanceMethod::Cosine,
        None,
    );
    println!("Distance: {:?}", d);
}
