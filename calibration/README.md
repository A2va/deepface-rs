# Calibration

This folder contains the scripts used to compute the **thresholds** and **confidence model parameters** used in `verify.rs`.

These scripts are based on the work of **Sefik Ilkin Serengil**, author of the DeepFace Python library, specifically:

* **Fine Tuning The Threshold in Face Recognition**
  [https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/)

* **Distance to Confidence** (logistic regression over similarity distances)
  [https://github.com/serengil/deepface/blob/master/experiments/distance-to-confidence.ipynb](https://github.com/serengil/deepface/blob/master/experiments/distance-to-confidence.ipynb)

The goal of these scripts is to reproduce DeepFace-style calibration for the distance metrics used by each recognition model.

---

## How to reproduce the calibration values

1. **Download the dataset**
   Download the dataset from [*here*](https://github.com/user-attachments/files/23835723/dataset.zip) and place it in the project root.
   The dataset must contain:

   * one folder per person, containing their images
   * a `master.csv` file indexing all images

2. **Generate distance CSVs for a given model**
   Run the Rust utility for the model you want to calibrate (currently **DeepID** and **FaceNet512** are supported):

   ```bash
   cargo run -p generate-model-distance facenet512
   ```

   This produces a CSV file named after the model, containing the pairwise distances for:

   * cosine
   * euclidean
   * euclidean L2
   * angular (if applicable)

3. **Compute thresholds and confidence parameters**

   Then run:

   ```bash
   uv run compute-thresholds.py facenet512.csv
   ```

   Example output:

   ```
   ❯ uv run threshol-conf.py --info facenet512.csv

   ==== Thresholds (copy into deepface-rs) ====

   ModelThreshold {
       cosine: 0.375397,
       euclidean: 17.347393,
       euclidean_l2: 0.834589,
       angular: 0.0,
   },

   ==== Confidence Models (copy into deepface-rs) ====

   DistanceMethod::Cosine => ModelConfidence {
       w: -7.073225,
       b: 1.565977,
       normalizer: 1.316728,
       denorm_max_true: 75.846307,
       denorm_min_true: 46.491800,
       denorm_max_false: 33.434419,
       denorm_min_false: 0.404086,
   },
   ...
   ```

These blocks can be copied directly into `verify.rs`.

---

## What the scripts do

### Threshold estimation

Thresholds are computed by analyzing the distribution of:

* **intra-class distances** (same person)
* **inter-class distances** (different persons)

The optimal threshold is chosen at the “tipping point” where false positives and false negatives balance according to the DeepFace methodology.

### Confidence model (logistic regression)

The confidence constants are fitted using a logistic regression model that maps **distance → probability of being the same person**.

For each distance method:

* `w` and `b` come from the logistic regression fit.
* `normalizer` rescales the distances.
* `denorm_*` values store the min/max true/false distances before normalization.

This reproduces the behavior of DeepFace’s confidence scoring.

---

## Custom datasets

You can use your own dataset.
Simply create a folder containing subfolders—one per identity—and place any images you want in each of these subfolders.

Then regenerate the `master.csv` with:

```bash
uv run generate-csv.py
```
