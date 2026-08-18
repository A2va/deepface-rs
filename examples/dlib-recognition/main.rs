use burn::Tensor;

use image::DynamicImage;

use deepface::detection::{Detector, NmsOptions, Yunet};
use deepface::metrics::{verify, DistanceMethod};
use deepface::recognition::{DlibRecognition, RecognitionModel, Recognizer};
use deepface::DlibDetectorModel;

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

    let model: DlibRecognition = DlibRecognition::new(DlibDetectorModel::Hog);
    model.embed(&img, result.clone(), None)
}

fn main() {
    let img = image::open("dataset2/img1.jpg").unwrap();
    let result1 = embed(img);

    let img = image::open("dataset2/img2.jpg").unwrap();
    let result2 = embed(img);

    let d = verify(
        result1,
        result2,
        RecognitionModel::DlibRecognition,
        DistanceMethod::Cosine,
        None,
    );
    println!("Distance: {:?}", d);
}
