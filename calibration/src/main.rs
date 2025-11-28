use anyhow::Context;

use burn::prelude::Backend;
use burn::{backend::NdArray, Tensor};
use deepface::ImageToTensor;
use image::{DynamicImage, GenericImageView};

use deepface::detection::{Detector, Yunet};
use deepface::recognition::{verify, DistanceMethod, NormalizationMethod, RecognitionModel};
use deepface::recognition::{DeepID, FaceNet512, Recognizer};

use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::{env, process};

enum AnyModel<B: Backend> {
    DeepID(DeepID<B>),
    FaceNet512(FaceNet512<B>),
}

impl<B: Backend<FloatElem = f32>> Recognizer<B> for AnyModel<B> {
    const SHAPE: (u32, u32) = (0, 0); // Can’t have a true constant here

    fn embed<I: ImageToTensor<B>>(
        &self,
        input: &I,
        norm: Option<NormalizationMethod>,
    ) -> Tensor<B, 1> {
        match self {
            AnyModel::DeepID(m) => m.embed(input, norm),
            AnyModel::FaceNet512(m) => m.embed(input, norm),
        }
    }
}

fn get_model<B: Backend<FloatElem = f32>>(name: &str) -> AnyModel<B> {
    match name {
        "deepid" => AnyModel::DeepID(DeepID::new()),
        "facenet512" => AnyModel::FaceNet512(FaceNet512::new()),
        _ => panic!("Unknown model {name}"),
    }
}

fn embed(
    img: DynamicImage,
    detector: &impl Detector<NdArray>,
    model: &AnyModel<NdArray>,
) -> Tensor<NdArray, 1> {
    let results = detector.detect(&img, 0.8, None);
    let results = results.first().unwrap();

    let subimg = img.view(results.x, results.y, results.w, results.h);
    model.embed(&subimg, Some(NormalizationMethod::ZeroOne))
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
    // headers.push_fied("angular"); // not ready

    let file = File::create(format!("{model_name}.csv"))?;
    let mut wtr = csv::Writer::from_writer(file);
    wtr.write_record(&headers)?;

    let detector: Yunet<NdArray> = Yunet::new();
    let model = get_model(&model_name);

    let rec_model_enum = match model_name.as_str() {
        "deepid" => RecognitionModel::DeepID,
        "facenet512" | "facenet" => RecognitionModel::FaceNet512,
        _ => panic!("not valid model"), // fallback (shouldn't happen due to get_model)
    };

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

        let cosine = verify(
            emb1.clone(),
            emb2.clone(),
            rec_model_enum,
            DistanceMethod::Cosine,
            None,
        )
        .distance;
        let euclid = verify(
            emb1.clone(),
            emb2.clone(),
            rec_model_enum,
            DistanceMethod::Euclidean,
            None,
        )
        .distance;
        let euclid_l2 = verify(
            emb1.clone(),
            emb2.clone(),
            rec_model_enum,
            DistanceMethod::EuclideanL2,
            None,
        )
        .distance;

        record.push_field(&cosine.to_string());
        record.push_field(&euclid.to_string());
        record.push_field(&euclid_l2.to_string());

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
