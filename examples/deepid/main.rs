use burn::{backend::NdArray, Tensor};
use image::{DynamicImage, GenericImage};

use deepface::detection::{Detector, Yunet};
use deepface::recognition::{verify, DistanceMethod, NormalizationMethod, RecognitionModel};
use deepface::recognition::{DeepID, Recognizer};

fn crop(mut img: DynamicImage) -> Tensor<NdArray, 1> {
    let model: Yunet<NdArray> = Yunet::new();
    let results = model.detect(img.clone(), 0.8, None);
    let results = results.first().unwrap();

    let subimg = img.sub_image(results.x, results.y, results.w, results.h);
    let subimg: DynamicImage = subimg.to_image().into();

    let model: DeepID<NdArray> = DeepID::new();
    model.embed(subimg, None)
}

fn main() {
    let img = image::open("dataset/img1.png").unwrap();
    let result1 = crop(img);

    let img = image::open("dataset/img2.png").unwrap();
    let result2 = crop(img);

    let d = verify(
        result1,
        result2,
        RecognitionModel::FaceNet512,
        DistanceMethod::Cosine,
    );

    println!("Distance: {:?}", d);
}
