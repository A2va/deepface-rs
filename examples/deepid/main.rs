use burn::Tensor;
use image::{DynamicImage, GenericImage};

use deepface::detection::{Detector, NmsOptions, Yunet};
use deepface::metrics::{verify, DistanceMethod};
use deepface::recognition::{DeepID, RecognitionModel, Recognizer};

fn embed(img: DynamicImage) -> Tensor<1> {
    let model: Yunet = Yunet::new();
    let results = model.detect(
        &img,
        NmsOptions {
            score_threshold: 0.8,
            ..NmsOptions::default()
        },
    );
    let result = results.first().unwrap();

    let model: DeepID = DeepID::new();
    model.embed(&img, *result, None)
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
