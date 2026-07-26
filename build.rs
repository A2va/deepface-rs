use burn_onnx::{LoadStrategy, ModelGen};

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
        .load_strategy(LoadStrategy::Embedded)
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

    const WEIGHTS: [(&'static str, &'static str); 4] = [
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/centerface.onnx",
            "centerface.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/yunet.onnx",
            "yunet.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/scrfd_10g_kps.onnx",
            "scrfd_10g.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/scrfd_500m_kps.onnx",
            "scrfd_500m.onnx",
        ),
    ];

    for (url, filename) in WEIGHTS {
        let filename = Path::new(filename);
        let feature = filename.file_stem().unwrap().to_str().unwrap().to_string();
        let feature = format!("CARGO_FEATURE_{}", feature.replace('-', "_").to_uppercase());

        if std::env::var(&feature).is_ok() {
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
}

fn recognition_models() {
    let path = Path::new("models/recognition");
    std::fs::create_dir_all(path).unwrap();
    const WEIGHTS: [(&'static str, &'static str); 4] = [
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/deepid.onnx",
            "deepid.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/facenet512.onnx",
            "facenet512.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/adaface_ir_18.onnx",
            "adaface_ir18.onnx",
        ),
        (
            "https://github.com/A2va/deepface-rs/releases/download/v0.0/adaface_ir_101.onnx",
            "adaface_ir101.onnx",
        ),
    ];
    for (url, filename) in WEIGHTS {
        let filename = Path::new(filename);
        let feature = filename.file_stem().unwrap().to_str().unwrap().to_string();
        let feature = format!("CARGO_FEATURE_{}", feature.replace('-', "_").to_uppercase());

        if std::env::var(&feature).is_ok() {
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
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DETECTION_CENTERFACE");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DETECTION_YUNET");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_SCRFD_10G");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_SCRFD_500M");

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_FACENET512");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_DEEPID");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_ADAFACE_IR18");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RECOGNITION_ADAFACE_IR101");

    test_files();
    detection_models();
    recognition_models();
}
