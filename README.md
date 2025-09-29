A rust implementation of the [deepface](https://github.com/serengil/deepface) python library.

# Supported models

Detction:
* Centerface
* Yunet

Recognition:
* DeepID
* Facenet512

To use one of these model, you must add them as features in your `Cargo.toml`:
```toml
deepface = {git = "https://github.com/A2va/deepface-rs", features = ["yunet", "facenet512"]}
```

