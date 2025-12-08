use burn::{backend::NdArray, Tensor};
use deepface::detection::dlib::DlibDetectorModel;
use image::{DynamicImage, GenericImage};

use deepface::detection::{Detector, Yunet};
use deepface::recognition::{verify, DistanceMethod, RecognitionModel};
use deepface::recognition::{DlibRecognition, Recognizer};

fn embed(mut img: DynamicImage) -> Tensor<NdArray, 1> {
    let model: DlibRecognition<NdArray> = DlibRecognition::new(DlibDetectorModel::Hog);
    model.embed(&img, None)
}

fn main() {
    let img = image::open("dataset2/kanna1.jpg").unwrap();
    let result1 = embed(img);

    let img = image::open("dataset2/kudo.jpg").unwrap();
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
