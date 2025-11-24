use burn::prelude::{Backend, ElementConversion, Tensor, ToElement};
use burn::tensor::linalg::{cosine_similarity, l2_norm, DEFAULT_EPSILON};

#[cfg(feature = "deepid")]
use crate::recognition::RecognitionModel;

/// All distance methods supported in [`similarity`].
#[derive(Clone, Copy)]
pub enum DistanceMethod {
    Cosine,
    Euclidean,
    EuclideanL2,
    Angular,
}

/// Output of [`verify`].
#[derive(Clone, Copy, Debug)]
pub struct VerifyResult {
    /// Confidence score indicating the likelihood that the images
    /// represent the same person. The score is between 0 and 100, where higher values
    /// indicate greater confidence in the verification result.
    pub confidence: f32,
    /// The distance measure between the face vectors.
    /// A lower distance indicates higher similarity.
    pub distance: f32,
    /// Indicates whether the images represent the same person, true if that's the case.
    /// This is true if the similarity is less or equal than the treshold for that model or given in [`verify`] input.
    pub verified: bool,
}

struct ModelThreshold {
    cosine: f32,
    euclidean: f32,
    euclidean_l2: f32,
    angular: f32,
}

struct ModelConfidence {
    w: f32,
    b: f32,
    normalizer: f32,
    denorm_max_true: f32,
    denorm_min_true: f32,
    denorm_max_false: f32,
    denorm_min_false: f32,
}

impl RecognitionModel {
    fn thresholds(&self) -> ModelThreshold {
        match self {
            #[cfg(feature = "deepid")]
            RecognitionModel::DeepID => ModelThreshold {
                cosine: 0.0044,
                euclidean: 27.9795,
                euclidean_l2: 0.1075,
                angular: 0.0,
            },
            #[cfg(feature = "facenet512")]
            RecognitionModel::FaceNet512 => ModelThreshold {
                cosine: 0.3754,
                euclidean: 17.3474,
                euclidean_l2: 0.8346,
                angular: 0.0,
            },
        }
    }

    fn confidence(&self, distance: DistanceMethod) -> ModelConfidence {
        match self {
            #[cfg(feature = "deepid")]
            RecognitionModel::DeepID => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -0.055115,
                    b: -6.119913,
                    normalizer: 0.251455,
                    denorm_max_true: 0.219329,
                    denorm_min_true: 0.219329,
                    denorm_max_false: 0.219329,
                    denorm_min_false: 0.216370,
                },
                DistanceMethod::Euclidean => ModelConfidence {
                    w: -0.257771,
                    b: -6.025072,
                    normalizer: 196.549220,
                    denorm_max_true: 0.232486,
                    denorm_min_true: 0.232486,
                    denorm_max_false: 0.232132,
                    denorm_min_false: 0.186460,
                },
                DistanceMethod::EuclideanL2 => ModelConfidence {
                    w: -0.221582,
                    b: -6.056230,
                    normalizer: 0.709161,
                    denorm_max_true: 0.228974,
                    denorm_min_true: 0.228974,
                    denorm_max_false: 0.228943,
                    denorm_min_false: 0.199848,
                },
                DistanceMethod::Angular => todo!("missing acos support in burn"),
            },
            #[cfg(feature = "facenet512")]
            RecognitionModel::FaceNet512 => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -7.073225,
                    b: 1.565977,
                    normalizer: 1.316728,
                    denorm_max_true: 75.846307,
                    denorm_min_true: 46.491800,
                    denorm_max_false: 33.434419,
                    denorm_min_false: 0.404086,
                },
                DistanceMethod::Euclidean => ModelConfidence {
                    w: -7.353721,
                    b: 2.659385,
                    normalizer: 36.887860,
                    denorm_max_true: 73.250955,
                    denorm_min_true: 34.998241,
                    denorm_max_false: 28.397737,
                    denorm_min_false: 0.906403,
                },
                DistanceMethod::EuclideanL2 => ModelConfidence {
                    w: -7.407328,
                    b: 3.068394,
                    normalizer: 1.622793,
                    denorm_max_true: 77.898013,
                    denorm_min_true: 36.123523,
                    denorm_max_false: 24.719646,
                    denorm_min_false: 1.288232,
                },
                DistanceMethod::Angular => todo!("missing acos support in burn"),
            },
        }
    }
}

// TODO Implement batching
fn angular_distance<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.unwrap_or_else(|| B::FloatElem::from_elem(DEFAULT_EPSILON));

    // Convert negative dimension to positive
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;

    // Compute dot product: sum(x1 * x2) along the specified dimension
    let dot_product = (x1.clone() * x2.clone()).sum_dim(dim_idx);

    // Compute L2 norms: ||x1|| and ||x2||
    let norm_x1 = l2_norm(x1, dim_idx);
    let norm_x2 = l2_norm(x2, dim_idx);

    // Calculate the denominator (product of the norms) with epsilon to avoid division by zero
    let denominator = norm_x1.clamp_min(eps) * norm_x2.clamp_min(eps);

    let similarity = dot_product / denominator;
    // np.arccos(similarity) / np.pi
    todo!("TODO Wait for burn to have arccos")
}

fn euclidean_similarity<B: Backend, const D: usize>(
    emb1: Tensor<B, D>,
    emb2: Tensor<B, D>,
    dim: i32,
) -> Tensor<B, D> {
    // Convert negative dimension to positive
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;

    let t = emb1 - emb2;
    l2_norm(t, dim_idx)
}

fn l2_normalize<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.unwrap_or_else(|| B::FloatElem::from_elem(DEFAULT_EPSILON));
    // Convert negative dimension to positive
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;

    let norm = l2_norm(x.clone(), dim_idx);
    x / (norm + eps)
}

fn confidence(
    distance: f32,
    model: RecognitionModel,
    method: DistanceMethod,
    verified: bool,
) -> f32 {
    let config = model.confidence(method);

    let distance = if config.normalizer > 1.0 {
        distance / config.normalizer
    } else {
        distance
    };

    let z = config.w * distance + config.b;
    let confidence = 100.0 * (1.0 / (1.0 + (-z).exp()));

    let (min_original, max_original, min_target, max_target) = match verified {
        true => (
            config.denorm_min_true,
            config.denorm_max_true,
            config.denorm_min_true.max(51.0),
            100.0,
        ),
        false => (
            config.denorm_min_false,
            config.denorm_max_false,
            0.0,
            config.denorm_max_false.max(49.0),
        ),
    };

    let confidence_distributed = ((confidence - min_original) / (max_original - min_original))
        * (max_target - min_target)
        + min_target;

    let confidence_distributed = if verified && (confidence_distributed < 51.0) {
        51.0f32
    } else if !verified && (confidence_distributed > 49.0) {
        49.0f32
    } else {
        confidence_distributed
    };

    let confidence_distributed = confidence_distributed.clamp(0.0, 100.0);
    (confidence_distributed * 100.0).round() / 100.0
}

/// Compare two emdeddings tensor with the specified method and return the distance between them.
/// Make sure that the two embeddings have been generated by the same model.
pub fn similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    method: DistanceMethod,
) -> Tensor<B, D> {
    match method {
        DistanceMethod::Cosine => 1 - cosine_similarity(x1, x2, -1, None),
        DistanceMethod::Euclidean => euclidean_similarity(x1, x2, -1),
        DistanceMethod::EuclideanL2 => {
            let x1_norm = l2_normalize(x1, -1, None);
            let x2_norm = l2_normalize(x2, -1, None);
            euclidean_similarity(x1_norm, x2_norm, -1)
        }
        DistanceMethod::Angular => angular_distance(x1, x2, -1, None),
    }
}

/// Verify that two face embeddings are the same or not.
/// If the threshold is None it will compare to the internal threshold value.
pub fn verify<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    model: RecognitionModel,
    method: DistanceMethod,
    threshold: Option<f32>,
) -> VerifyResult {
    let pretuned_threshold = model.thresholds();

    let threshold = threshold.unwrap_or_else(|| match method {
        DistanceMethod::Cosine => pretuned_threshold.cosine,
        DistanceMethod::Euclidean => pretuned_threshold.euclidean,
        DistanceMethod::EuclideanL2 => pretuned_threshold.euclidean_l2,
        DistanceMethod::Angular => pretuned_threshold.angular,
    });

    let distance = similarity(x1, x2, method.clone()).into_scalar().to_f32();
    let verified = distance <= threshold;
    let confidence = confidence(distance, model, method, verified);

    VerifyResult {
        distance,
        confidence: confidence,
        verified: verified,
    }
}
