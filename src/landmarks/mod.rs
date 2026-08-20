//! Model-agnostic facial landmarks.
//!
//! Every landmark model (the 5-point detector outputs, dlib's 68 points, ...)
//! is normalised onto the same **canonical space** — the MediaPipe 478 model.
//! A [`Landmarks`] value stores a model's raw points plus a static
//! `raw index → canonical index` table ([`ModelKind`]).
//!
//! Consumers never need to know a model's exact indices: [`Landmarks::part`]
//! and [`Landmarks::center`] query semantic regions ([`FacePart`]) best-effort.

use crate::ImageToTensor;

/// A semantic region of the face, expressed as a set of canonical (MediaPipe 478)
/// indices. Model-specific indices never leak into the query API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacePart {
    /// Person's right eye (image left).
    RightEye,
    /// Person's left eye (image right).
    LeftEye,
    /// Person's right eyebrow.
    RightBrow,
    /// Person's left eyebrow.
    LeftBrow,
    /// Nose bridge and tip.
    NoseBridge,
    /// Nose tip.
    NoseTip,
    /// Jaw / face contour.
    Jaw,
    /// Outer mouth outline.
    Mouth,
    /// Inner mouth outline.
    MouthInner,
    /// Person's right mouth corner.
    RightMouth,
    /// Person's left mouth corner.
    LeftMouth,
}

impl FacePart {
    /// Canonical (MediaPipe 478) indices covered by this part.
    fn canonical(&self) -> &'static [usize] {
        match self {
            FacePart::RightEye => &[33, 160, 158, 133, 153, 144],
            FacePart::LeftEye => &[362, 385, 387, 263, 373, 380],
            FacePart::RightBrow => &[70, 63, 105, 66, 107],
            FacePart::LeftBrow => &[336, 296, 334, 293, 300],
            FacePart::NoseBridge => &[168, 197, 5, 4, 98, 97, 2, 326, 327],
            FacePart::NoseTip => &[4],
            FacePart::Jaw => &[
                162, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 389,
            ],
            FacePart::Mouth => &[61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181],
            FacePart::MouthInner => &[78, 82, 13, 312, 308, 317, 14, 87],
            FacePart::RightMouth => &[61],
            FacePart::LeftMouth => &[291],
        }
    }
}

/// Which landmark model produced a [`Landmarks`] value. Each variant owns the
/// static `raw index → canonical index` table for that model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// The 5-point canonical output shared by SCRFD / YuNet / CenterFace.
    Five,
    /// Dlib's 68-point model.
    #[cfg(feature = "dlib-detection")]
    Dlib68,
}

/// The 5-point detectors output canonical points in ArcFace order:
/// right eye, left eye, nose, right mouth corner, left mouth corner.
const FIVE_TO_CANONICAL: [usize; 5] = [33, 263, 4, 61, 291];

impl ModelKind {
    /// The `raw index → canonical index` table for this model.
    pub fn to_canonical(&self) -> &'static [usize] {
        match self {
            ModelKind::Five => &FIVE_TO_CANONICAL,
            #[cfg(feature = "dlib-detection")]
            ModelKind::Dlib68 => &dlib::DLIB_TO_CANONICAL,
        }
    }
}

/// Facial landmark points in a model's native ordering, with enough metadata to
/// answer semantic queries without knowing the model's indices.
#[derive(Clone, Debug)]
pub struct Landmarks {
    points: Vec<(f32, f32)>,
    kind: ModelKind,
}

impl Landmarks {
    /// Build landmarks from a model's raw points.
    pub fn new(points: Vec<(f32, f32)>, kind: ModelKind) -> Self {
        Self { points, kind }
    }

    /// The raw points in this model's native ordering.
    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }

    /// The model that produced these landmarks.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// The `raw index → canonical index` table, e.g. for head-pose estimation.
    pub fn canonical_indices(&self) -> &'static [usize] {
        self.kind.to_canonical()
    }

    /// Best-effort: the raw points belonging to `part`. Returns `None` when the
    /// model doesn't carry that region (e.g. `Five` has no jaw contour).
    pub fn part(&self, part: FacePart) -> Option<Vec<(f32, f32)>> {
        let canonical = part.canonical();
        let hits: Vec<(f32, f32)> = self
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| canonical.contains(&self.kind.to_canonical()[*i]))
            .map(|(_, p)| *p)
            .collect();
        if hits.is_empty() {
            None
        } else {
            Some(hits)
        }
    }

    /// Best-effort: the mean of `part`'s points (e.g. the eye center from its
    /// outline). Falls back to a single point when the model has no outline.
    pub fn center(&self, part: FacePart) -> Option<(f32, f32)> {
        let pts = self.part(part)?;
        let n = pts.len() as f32;
        Some((
            pts.iter().map(|p| p.0).sum::<f32>() / n,
            pts.iter().map(|p| p.1).sum::<f32>() / n,
        ))
    }

    /// The five ArcFace alignment points in canonical order:
    /// `[right eye, left eye, nose, right mouth, left mouth]`.
    pub fn to_5_points(&self) -> Option<[(f32, f32); 5]> {
        Some([
            self.center(FacePart::RightEye)?,
            self.center(FacePart::LeftEye)?,
            self.center(FacePart::NoseTip)?,
            self.center(FacePart::RightMouth)?,
            self.center(FacePart::LeftMouth)?,
        ])
    }

    /// Default padding: 20% of the bounding box's largest side.
    const DEFAULT_PADDING_RATIO: f32 = 0.2;

    /// Convert these landmarks into a [`crate::detection::FacialAreaRegion`],
    /// deriving the bounding box from the points and expanding it by a default
    /// padding (20% of the largest side). Confidence is unknown (`None`).
    pub fn to_facial_area_region(&self) -> crate::detection::FacialAreaRegion {
        let (min_x, min_y, max_x, max_y) = self.bounds();
        let padding = ((max_x - min_x).max(max_y - min_y) * Self::DEFAULT_PADDING_RATIO) as u32;
        self.to_facial_area_region_with_padding(padding)
    }

    /// Convert these landmarks into a [`crate::detection::FacialAreaRegion`],
    /// deriving the bounding box from the points and expanding it by `padding`
    /// on each side. Confidence is unknown (`None`).
    fn to_facial_area_region_with_padding(
        &self,
        padding: u32,
    ) -> crate::detection::FacialAreaRegion {
        let (min_x, min_y, max_x, max_y) = self.bounds();
        let p = padding as f32;
        crate::detection::FacialAreaRegion {
            x: (min_x - p).max(0.0) as u32,
            y: (min_y - p).max(0.0) as u32,
            w: (max_x - min_x + 2.0 * p) as u32,
            h: (max_y - min_y + 2.0 * p) as u32,
            confidence: None,
            landmarks: Some(self.clone()),
        }
    }

    fn bounds(&self) -> (f32, f32, f32, f32) {
        let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &self.points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        (min_x, min_y, max_x, max_y)
    }
}

/// A standalone facial landmark model. Unlike [`crate::detection::Detector`],
/// it predicts landmarks only — but may run its own face detection internally
/// (e.g. MediaPipe, dlib).
pub trait Landmarker {
    /// Predict landmarks for every face in the input image.
    fn predict<I: ImageToTensor>(&self, input: &I) -> Vec<Landmarks>;
}

#[cfg(feature = "dlib-detection")]
pub mod dlib;
#[cfg(feature = "dlib-detection")]
pub use dlib::DlibLandmarks;

#[cfg(test)]
mod tests {
    use super::{Landmarks, ModelKind};

    #[test]
    fn five_points_to_facial_area_region() {
        let lm = Landmarks::new(
            vec![
                (10.0, 20.0),
                (50.0, 20.0),
                (30.0, 40.0),
                (10.0, 60.0),
                (50.0, 60.0),
            ],
            ModelKind::Five,
        );

        let region = lm.to_facial_area_region_with_padding(5);
        assert_eq!(region.x, 5);
        assert_eq!(region.y, 15);
        assert_eq!(region.w, 50);
        assert_eq!(region.h, 50);
        assert_eq!(region.confidence, None);
        assert!(region.landmarks.is_some());

        // Default padding = 20% of the largest side (40 → 8).
        let region = lm.to_facial_area_region();
        assert_eq!(region.x, 2);
        assert_eq!(region.y, 12);
        assert_eq!(region.w, 56);
        assert_eq!(region.h, 56);

        let eye = lm.center(super::FacePart::RightEye).unwrap();
        assert_eq!(eye, (10.0, 20.0));
        let five = lm.to_5_points().unwrap();
        assert_eq!(five[0], (10.0, 20.0));
        assert_eq!(five[1], (50.0, 20.0));
    }
}
