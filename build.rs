use burn_import::onnx::{ModelGen, RecordType};

use std::path::Path;

fn download_file_if_necessary<P: AsRef<Path>>(url: &str, path: P) {
    let path = path.as_ref();
    if path.exists() {
        return;
    }

    println!("url {}", url);

    let response = reqwest::blocking::get(url).unwrap();
    let bytes = response.bytes().unwrap();

    std::fs::write(path, bytes).unwrap();
}

fn burn_onnx_converter<P: AsRef<Path>>(path: P, out_dir: &str) {
    ModelGen::new()
        .input(path.as_ref().to_str().unwrap())
        .out_dir(out_dir)
        .record_type(RecordType::Bincode)
        .embed_states(false)
        .run_from_script();
}

fn test_files() {
    let path = Path::new("dataset");
    std::fs::create_dir_all(path).unwrap();

    let url = "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/img1.jpg";
    download_file_if_necessary(url, path.join("one_face.jpg"));
}

fn detection_models() {
    let path = Path::new("models/detection");
    std::fs::create_dir_all(path).unwrap();

    const WEIGHTS: [(&'static str, &'static str); 2] = [
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/centerface.onnx",
            "centerface.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/yunet.onnx",
            "yunet.onnx",
        ),
    ];

    for (url, filename) in WEIGHTS {
        let file = path.join(filename);
        println!("url {}", url);
        download_file_if_necessary(url, &file);

        if file.exists() {
            let extension = file.extension().unwrap().to_str().unwrap();
            match extension {
                "onnx" => burn_onnx_converter(file, path.to_str().unwrap()),
                _ => (),
            }
        }
    }
}

fn recognition_models() {
    let path = Path::new("models/recognition");
    std::fs::create_dir_all(path).unwrap();
    const WEIGHTS: [(&'static str, &'static str); 2] = [
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/deepid.onnx",
            "deepid.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/facenet512.onnx",
            "facenet512.onnx",
        ),
    ];
    for (url, filename) in WEIGHTS {
        let file = path.join(filename);
        download_file_if_necessary(url, &file);

        if file.exists() {
            println!("file exists: {}", file.display());
            let extension = file.extension().unwrap().to_str().unwrap();
            match extension {
                "onnx" => burn_onnx_converter(file, path.to_str().unwrap()),
                _ => (),
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    test_files();
    detection_models();
    recognition_models();
}
