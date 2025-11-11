use burn::prelude::{Backend, ElementConversion, Tensor, ToElement};
use burn::tensor::linalg::{cosine_similarity, l2_norm, DEFAULT_EPSILON};

#[cfg(feature = "deepid")]
use crate::recognition::RecognitionModel;

/// All distance methods supported in [`similarity`]
#[derive(Clone, Copy)]
pub enum DistanceMethod {
    Cosine,
    Euclidian,
    EuclidianL2,
    Angular,
}

#[derive(Clone, Copy, Debug)]
pub struct VerifyResult {
    confidence: f32,
    similarity: f32,
    verified: bool,
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

// TODO: For now the thresholds and confidence value are directly based on the deepface repo but it would be great to recompute them.
// But it will require to have python bindings.

impl RecognitionModel {
    fn thresholds(&self) -> ModelThreshold {
        match self {
            #[cfg(feature = "deepid")]
            RecognitionModel::DeepID => ModelThreshold {
                cosine: 0.015,
                euclidean: 45.0,
                euclidean_l2: 0.17,
                angular: 0.04,
            },
            #[cfg(feature = "facenet512")]
            RecognitionModel::FaceNet512 => ModelThreshold {
                cosine: 0.35,
                euclidean: 23.56,
                euclidean_l2: 1.04,
                angular: 0.35,
            },
        }
    }

    fn confidence(&self, distance: DistanceMethod) -> ModelConfidence {
        match self {
            #[cfg(feature = "deepid")]
            RecognitionModel::DeepID => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -1.1109389867203003,
                    b: -1.644356005882629,
                    normalizer: 0.142463,
                    denorm_max_true: 16.114369377586183,
                    denorm_min_true: 15.966222060549578,
                    denorm_max_false: 15.96057368926546,
                    denorm_min_false: 14.153198282635255,
                },
                DistanceMethod::Euclidian => ModelConfidence {
                    w: -4.267900227772648,
                    b: -0.11777629548659044,
                    normalizer: 124.194689,
                    denorm_max_true: 27.2342405984107,
                    denorm_min_true: 16.08201206739819,
                    denorm_max_false: 15.90739039527823,
                    denorm_min_false: 1.230125408063358,
                },
                DistanceMethod::EuclidianL2 => ModelConfidence {
                    w: -3.6681410379067394,
                    b: -0.9230555193862335,
                    normalizer: 0.533784,
                    denorm_max_true: 21.681829939622233,
                    denorm_min_true: 17.57160935306287,
                    denorm_max_false: 17.548244679217202,
                    denorm_min_false: 5.309829288484883,
                },
                DistanceMethod::Angular => ModelConfidence {
                    w: -0.2707576106983799,
                    b: -3.87004749575164,
                    normalizer: 0.171993,
                    denorm_max_true: 2.02619851743467,
                    denorm_min_true: 2.0220784432439474,
                    denorm_max_false: 2.021498115793411,
                    denorm_min_false: 1.951974654804089,
                },
            },
            #[cfg(feature = "facenet512")]
            RecognitionModel::FaceNet512 => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -6.502269165856082,
                    b: 1.679048923097668,
                    normalizer: 1.206694,
                    denorm_max_true: 77.17253153662926,
                    denorm_min_true: 41.790002608273234,
                    denorm_max_false: 20.618350202170916,
                    denorm_min_false: 0.7976712344840693,
                },
                DistanceMethod::Euclidian => ModelConfidence {
                    w: -6.716177467853723,
                    b: 2.790978346203265,
                    normalizer: 18.735288,
                    denorm_max_true: 74.76412617567517,
                    denorm_min_true: 40.4423755909089,
                    denorm_max_false: 25.840858374979504,
                    denorm_min_false: 1.9356150486888306,
                },
                DistanceMethod::EuclidianL2 => ModelConfidence {
                    w: -6.708710331202137,
                    b: 2.9094193067398195,
                    normalizer: 1.553508,
                    denorm_max_true: 75.45756719896039,
                    denorm_min_true: 40.4509428022908,
                    denorm_max_false: 30.555931000001184,
                    denorm_min_false: 2.189644991619842,
                },
                DistanceMethod::Angular => ModelConfidence {
                    w: -6.371147050396505,
                    b: 0.6766460615182355,
                    normalizer: 0.56627,
                    denorm_max_true: 45.802357900723386,
                    denorm_min_true: 24.327312950719133,
                    denorm_max_false: 16.95267765757785,
                    denorm_min_false: 5.063533287198758,
                },
            },
        }
    }
}

// TODO Implement batching
fn angular_similarity<B: Backend, const D: usize>(
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

    let norm = l2_norm(x.clone(), dim_idx).unsqueeze_dim(dim_idx);
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

/// Compare two emdeddings tensor with the specified method.
/// Make sure that the two embeddings have been generated by the same model.
pub fn similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    method: DistanceMethod,
) -> Tensor<B, D> {
    match method {
        DistanceMethod::Cosine => 1 - cosine_similarity(x1, x2, -1, None),
        DistanceMethod::Euclidian => euclidean_similarity(x1, x2, -1),
        DistanceMethod::EuclidianL2 => {
            let x1_norm = l2_normalize(x1, -1, None);
            let x2_norm = l2_normalize(x2, -1, None);
            euclidean_similarity(x1_norm, x2_norm, -1)
        }
        DistanceMethod::Angular => angular_similarity(x1, x2, -1, None),
    }
}

pub fn verify<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    model: RecognitionModel,
    method: DistanceMethod,
) -> VerifyResult {
    let pretuned_threshold = model.thresholds();
    let threshold = match method {
        DistanceMethod::Cosine => pretuned_threshold.cosine,
        DistanceMethod::Euclidian => pretuned_threshold.euclidean,
        DistanceMethod::EuclidianL2 => pretuned_threshold.euclidean_l2,
        DistanceMethod::Angular => pretuned_threshold.angular,
    };

    let distance = similarity(x1, x2, method.clone()).into_scalar().to_f32();
    let verified = distance <= threshold;
    let confidence = confidence(distance, model, method, verified);

    VerifyResult {
        similarity: distance,
        confidence: confidence,
        verified: verified,
    }
}
