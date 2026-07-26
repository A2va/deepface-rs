use burn::Tensor;
use image::DynamicImage;

use deepface::detection::{Detector, NmsOptions, Scrfd10g};
use deepface::metrics::{verify, DistanceMethod};
use deepface::recognition::{AdaFaceIR101, RecognitionModel, Recognizer};

fn embed(img: DynamicImage) -> Tensor<1> {
    let model: Scrfd10g = Scrfd10g::new();
    let results = model.detect(
        &img,
        NmsOptions {
            score_threshold: 0.7,
            ..NmsOptions::default()
        },
    );
    let result = results.first().unwrap();

    let model: AdaFaceIR101 = AdaFaceIR101::new();
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
        RecognitionModel::FaceNet512,
        DistanceMethod::Cosine,
        None,
    );
    println!("Distance: {:?}", d);
}
