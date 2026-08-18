use anyhow::Context;
use burn::Tensor;
use image::DynamicImage;

use deepface::detection::{Detector, FacialAreaRegion, NmsOptions, Yunet};
use deepface::recognition::{
    AdaFaceIR101, AdaFaceIR18, DeepID, DlibRecognition, FaceNet512, Recognizer,
};

use deepface::metrics::{distance, DistanceMethod};
use deepface::recognition::NormalizationMethod;
use deepface::DlibDetectorModel;
use deepface::ImageToTensor;

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::{env, process};

enum AnyModel {
    DeepID(DeepID),
    FaceNet512(FaceNet512),
    DlibRecognition(DlibRecognition),
    AdaFaceIR18(AdaFaceIR18),
    AdaFaceIR101(AdaFaceIR101),
}

impl Recognizer for AnyModel {
    fn embed<I: ImageToTensor>(
        &self,
        input: &I,
        face: FacialAreaRegion,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<1> {
        match self {
            AnyModel::DeepID(m) => m.embed(input, face, norm),
            AnyModel::FaceNet512(m) => m.embed(input, face, norm),
            AnyModel::DlibRecognition(m) => m.embed(input, face, norm),
            AnyModel::AdaFaceIR18(m) => m.embed(input, face, norm),
            AnyModel::AdaFaceIR101(m) => m.embed(input, face, norm),
        }
    }
}

fn get_model(name: &str) -> AnyModel {
    match name {
        "deepid" => AnyModel::DeepID(DeepID::new()),
        "facenet512" => AnyModel::FaceNet512(FaceNet512::new()),
        "dlib-recognition" => {
            AnyModel::DlibRecognition(DlibRecognition::new(DlibDetectorModel::Cnn))
        }
        "adaface-ir18" => AnyModel::AdaFaceIR18(AdaFaceIR18::new()),
        "adaface-ir101" => AnyModel::AdaFaceIR101(AdaFaceIR101::new()),
        _ => panic!("Unknown model {name}"),
    }
}

fn embed(img: DynamicImage, detector: &impl Detector, model: &AnyModel) -> Tensor<1> {
    let results = detector.detect(
        &img,
        NmsOptions {
            score_threshold: 0.8,
            ..NmsOptions::default()
        },
    );
    let result = results.first().unwrap();
    model.embed(&img, result.clone(), Some(NormalizationMethod::ZeroOne))
}

fn generate_distance_csv(model_name: &str) -> Result<(), Box<dyn Error>> {
    let model_name = model_name.to_lowercase();

    let file = File::open("dataset/master.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    // Add distance headers
    let mut headers = rdr.headers()?.clone();
    headers.push_field("cosine");
    headers.push_field("euclidean");
    headers.push_field("euclidean_l2");
    headers.push_field("angular");

    let file = File::create(format!("{model_name}.csv"))?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&headers)?;

    let detector: Yunet = Yunet::new();
    let model = get_model(&model_name);

    for (i, result) in rdr.records().enumerate() {
        let mut record = result?;

        let img1_path = Path::new("dataset").join(record.get(0).unwrap());
        let img1 = image::open(&img1_path).context(format!(
            "Failed to open or process image file: {}",
            img1_path.display()
        ))?;
        let emb1 = embed(img1, &detector, &model);

        let img2_path = Path::new("dataset").join(record.get(1).unwrap());
        let img2 = image::open(&img2_path).context(format!(
            "Failed to open or process image file: {}",
            img2_path.display()
        ))?;
        let emb2 = embed(img2, &detector, &model);

        let cosine = distance(emb1.clone(), emb2.clone(), DistanceMethod::Cosine);
        let euclid = distance(emb1.clone(), emb2.clone(), DistanceMethod::Euclidean);
        let euclid_l2 = distance(emb1.clone(), emb2.clone(), DistanceMethod::EuclideanL2);
        let angular = distance(emb1.clone(), emb2.clone(), DistanceMethod::Angular);

        record.push_field(&cosine.to_string());
        record.push_field(&euclid.to_string());
        record.push_field(&euclid_l2.to_string());
        record.push_field(&angular.to_string());

        wtr.write_record(&record)?;
        println!("{i}");
    }
    wtr.flush()?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() == 2,
        "Usage: <program> <model_name>\nExample: generate-model-csv facenet512"
    );

    if let Err(err) = generate_distance_csv(&args[1]) {
        eprintln!("error running evaluation: {}", err);
        process::exit(1);
    }

    println!("Distance computation completed. Now run the Python script to generate thresholds and confidence values.");
}
