use burn::{backend::NdArray, Tensor};
use image::{DynamicImage, GenericImage};

use deepface::detection::{Detector, Yunet};
use deepface::recognition::{verify, DistanceMethod, RecognitionModel};
use deepface::recognition::{DeepID, Recognizer};

fn embed(mut img: DynamicImage) -> Tensor<NdArray, 1> {
    let model: Yunet<NdArray> = Yunet::new();
    let results = model.detect(&img, 0.8, None);
    let results = results.first().unwrap();

    let subimg = img.sub_image(results.x, results.y, results.w, results.h);

    let model: DeepID<NdArray> = DeepID::new();
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
        RecognitionModel::DeepID,
        DistanceMethod::Cosine,
        None,
    );
    println!("Distance: {:?}", d);
}
