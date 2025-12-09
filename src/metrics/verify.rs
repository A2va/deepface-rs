use burn::prelude::{Backend, Tensor};

use crate::metrics::distance::{distance, DistanceMethod};
use crate::recognition::RecognitionModel;

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

    let distance = distance(x1, x2, method.clone());
    let verified = distance <= threshold;
    let confidence = confidence(distance, model, method, verified);

    VerifyResult {
        distance: distance,
        confidence: confidence,
        verified: verified,
    }
}
