use super::DistanceMethod;
use crate::recognition::RecognitionModel;

pub(super) struct ModelThreshold {
    pub(super) cosine: f32,
    pub(super) euclidean: f32,
    pub(super) euclidean_l2: f32,
    pub(super) angular: f32,
}

pub(super) struct ModelConfidence {
    pub(super) w: f32,
    pub(super) b: f32,
    pub(super) normalizer: f32,
    pub(super) denorm_max_true: f32,
    pub(super) denorm_min_true: f32,
    pub(super) denorm_max_false: f32,
    pub(super) denorm_min_false: f32,
}

impl RecognitionModel {
    pub(super) fn thresholds(&self) -> ModelThreshold {
        match self {
            #[cfg(feature = "deepid")]
            RecognitionModel::DeepID => ModelThreshold {
                cosine: 0.0044,
                euclidean: 27.9795,
                euclidean_l2: 0.1075,
                angular: 0.029881,
            },
            #[cfg(feature = "facenet512")]
            RecognitionModel::FaceNet512 => ModelThreshold {
                cosine: 0.3754,
                euclidean: 17.3474,
                euclidean_l2: 0.8346,
                angular: 0.2755,
            },
            #[cfg(feature = "dlib-recognition")]
            RecognitionModel::DlibRecognition => ModelThreshold {
                cosine: 0.051966,
                euclidean: 0.506886,
                euclidean_l2: 0.321005,
                angular: 0.0,
            },
            #[cfg(feature = "adaface-ir18")]
            RecognitionModel::AdaFaceIR18 => ModelThreshold {
                cosine: 0.722152,
                euclidean: 1.160441,
                euclidean_l2: 1.160441,
                angular: 0.401766,
            },
            #[cfg(feature = "adaface-ir101")]
            RecognitionModel::AdaFaceIR101 => ModelThreshold {
                cosine: 0.710982,
                euclidean: 1.138921,
                euclidean_l2: 1.138921,
                angular: 0.395102,
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("no recognition model enabled"),
        }
    }

    pub(super) fn confidence(&self, distance: DistanceMethod) -> ModelConfidence {
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
                DistanceMethod::Angular => ModelConfidence {
                    w: -0.070299,
                    b: -6.116134,
                    normalizer: 0.230753,
                    denorm_max_true: 0.219750,
                    denorm_min_true: 0.219750,
                    denorm_max_false: 0.219747,
                    denorm_min_false: 0.216675,
                },
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
                DistanceMethod::Angular => ModelConfidence {
                    w: -7.152698,
                    b: 0.746744,
                    normalizer: 0.602584,
                    denorm_max_true: 45.973379,
                    denorm_min_true: 24.602752,
                    denorm_max_false: 19.465000,
                    denorm_min_false: 2.756226,
                },
            },
            #[cfg(feature = "dlib-recognition")]
            RecognitionModel::DlibRecognition => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -4.692252,
                    b: -1.661671,
                    normalizer: 0.267421,
                    denorm_max_true: 15.101393,
                    denorm_min_true: 13.398194,
                    denorm_max_false: 12.895675,
                    denorm_min_false: 5.134501,
                },
                DistanceMethod::Euclidean => ModelConfidence {
                    w: -7.434863,
                    b: 2.756446,
                    normalizer: 1.090731,
                    denorm_max_true: 72.362367,
                    denorm_min_true: 36.174570,
                    denorm_max_false: 19.444710,
                    denorm_min_false: 0.920814,
                },
                DistanceMethod::EuclideanL2 => ModelConfidence {
                    w: -7.342223,
                    b: 1.328352,
                    normalizer: 0.731329,
                    denorm_max_true: 52.653689,
                    denorm_min_true: 30.163954,
                    denorm_max_false: 25.704175,
                    denorm_min_false: 1.727231,
                },
                DistanceMethod::Angular => todo!("not supported"),
            },
            #[cfg(feature = "adaface-ir18")]
            RecognitionModel::AdaFaceIR18 => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -7.249017,
                    b: 2.586678,
                    normalizer: 1.181916,
                    denorm_max_true: 83.611532,
                    denorm_min_true: 32.058240,
                    denorm_max_false: 13.592328,
                    denorm_min_false: 0.935598,
                },
                DistanceMethod::Euclidean => ModelConfidence {
                    w: -7.599070,
                    b: 3.894672,
                    normalizer: 1.537476,
                    denorm_max_true: 75.648389,
                    denorm_min_true: 22.067086,
                    denorm_max_false: 11.405103,
                    denorm_min_false: 2.402369,
                },
                DistanceMethod::EuclideanL2 => ModelConfidence {
                    w: -7.599070,
                    b: 3.894672,
                    normalizer: 1.537476,
                    denorm_max_true: 75.648392,
                    denorm_min_true: 22.067088,
                    denorm_max_false: 11.405104,
                    denorm_min_false: 2.402370,
                },
                DistanceMethod::Angular => ModelConfidence {
                    w: -6.971584,
                    b: 0.846587,
                    normalizer: 0.558230,
                    denorm_max_true: 39.894881,
                    denorm_min_true: 16.952788,
                    denorm_max_false: 11.742187,
                    denorm_min_false: 4.542691,
                },
            },
            #[cfg(feature = "adaface-ir101")]
            RecognitionModel::AdaFaceIR101 => match distance {
                DistanceMethod::Cosine => ModelConfidence {
                    w: -7.019272,
                    b: 2.534465,
                    normalizer: 1.130832,
                    denorm_max_true: 84.652883,
                    denorm_min_true: 34.088487,
                    denorm_max_false: 7.577590,
                    denorm_min_false: 1.115327,
                },
                DistanceMethod::Euclidean => ModelConfidence {
                    w: -7.553636,
                    b: 3.872974,
                    normalizer: 1.503883,
                    denorm_max_true: 78.253007,
                    denorm_min_true: 22.753692,
                    denorm_max_false: 7.412923,
                    denorm_min_false: 2.458654,
                },
                DistanceMethod::EuclideanL2 => ModelConfidence {
                    w: -7.553636,
                    b: 3.872974,
                    normalizer: 1.503883,
                    denorm_max_true: 78.253009,
                    denorm_min_true: 22.753694,
                    denorm_max_false: 7.412924,
                    denorm_min_false: 2.458655,
                },
                DistanceMethod::Angular => ModelConfidence {
                    w: -7.327868,
                    b: 0.969183,
                    normalizer: 0.541765,
                    denorm_max_true: 43.818310,
                    denorm_min_true: 18.058241,
                    denorm_max_false: 9.518956,
                    denorm_min_false: 4.738981,
                },
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("no recognition model enabled"),
        }
    }
}
