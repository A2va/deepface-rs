use burn::{backend::NdArray, Tensor};
use deepface::detection::dlib::DlibDetectorModel;
use image::DynamicImage;

use deepface::metrics::{verify, DistanceMethod};
use deepface::recognition::{DlibRecognition, RecognitionModel, Recognizer};

fn embed(img: DynamicImage) -> Tensor<NdArray, 1> {
    let model: DlibRecognition<NdArray> = DlibRecognition::new(DlibDetectorModel::Hog);
    model.embed(&img, None)
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
