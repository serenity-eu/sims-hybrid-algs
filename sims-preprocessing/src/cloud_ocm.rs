//! OmniCloudMask (OCM) cloud-mask **inference** via ONNX Runtime (`ort`),
//! producing a georeferenced 3-class [`LabelRaster`] for [`crate::cloud`].
//!
//! Feature-gated behind `ocm` (pulls `ort` + `image`). This is the finetuned
//! detector we validated (2-model ONNX ensemble, mean-logit, softmax, tuned
//! confidence threshold τ = 0.75 on `P(cloud)+P(thin)`), matched to UP42's
//! `cloudCoverage` to a mean ~5.8 pp difference across 841 quicklooks.
//!
//! **Not wired into the preprocessing pipeline** — these are standalone APIs.
//!
//! OCM native classes are `0=clear, 1=thick-cloud, 2=thin-cloud, 3=shadow`; the
//! output is collapsed to [`crate::cloud`]'s convention `0=clear, 1=cloud,
//! 2=shadow` (with [`crate::cloud::NODATA`] for invalid pixels) so it feeds
//! [`crate::cloud::fragment_cloud_fractions`] directly.

use std::path::Path;

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::cloud::{LabelRaster, NODATA};

/// Tuned confidence threshold on `P(cloud) + P(thin-cloud)`: a pixel is cloud
/// iff the summed cloud softmax probability ≥ this. τ = 0.75 was the value that
/// made the coverage estimate unbiased vs UP42 `cloudCoverage`.
pub const CLOUD_TAU: f32 = 0.75;

const MODEL_FILES: [&str; 2] = ["ocm_regnety_004.onnx", "ocm_edgenext_small.onnx"];

/// A loaded OmniCloudMask ONNX ensemble ready for inference.
pub struct OcmEnsemble {
    sessions: Vec<Session>,
    tau: f32,
}

impl OcmEnsemble {
    /// Load the 2-model ensemble from a directory containing
    /// `ocm_regnety_004.onnx` and `ocm_edgenext_small.onnx`.
    pub fn load(model_dir: &Path) -> ort::Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut sessions = Vec::with_capacity(MODEL_FILES.len());
        for f in MODEL_FILES {
            sessions.push(
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(threads)?
                    .commit_from_file(model_dir.join(f))?,
            );
        }
        Ok(Self {
            sessions,
            tau: CLOUD_TAU,
        })
    }

    /// Override the cloud confidence threshold (default [`CLOUD_TAU`]).
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Predict a 3-class label map (`0=clear, 1=cloud, 2=shadow`;
    /// [`NODATA`] for invalid) from an RGBA image buffer (row-major `H*W*4` u8).
    ///
    /// **Validity = the footprint (`alpha > 0`).** Coverage is measured relative
    /// to the footprint, matching UP42's `cloudCoverage` convention (and the
    /// Python reference the metrics were validated against). Normalization
    /// statistics follow OCM's `no_data_value = 0` convention and are computed
    /// over non-black pixels only, but in-footprint black pixels are still
    /// classified (they come out clear/shadow), not dropped.
    ///
    /// Replicates OCM's preprocessing: per-channel z-score, zero-pad to a
    /// multiple of 32, mean-logit ensemble, softmax, then the tuned rule
    /// (`cloud` if `P(cloud)+P(thin) ≥ τ`, else `shadow` if argmax is shadow,
    /// else `clear`).
    pub fn predict_labels(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> ort::Result<Vec<u8>> {
        let (w, h) = (width, height);
        assert_eq!(rgba.len(), w * h * 4, "rgba buffer must be width*height*4");

        // ── normalization stats over non-black in-footprint pixels (OCM's
        //    all-zero nodata convention); `norm_valid` also decides the model input ──
        let mut norm_valid = vec![false; w * h];
        let (mut sum, mut sq) = ([0f64; 3], [0f64; 3]);
        let mut nv = 0usize;
        for i in 0..w * h {
            let (r, g, b, a) = (
                rgba[i * 4],
                rgba[i * 4 + 1],
                rgba[i * 4 + 2],
                rgba[i * 4 + 3],
            );
            let v = a > 0 && !(r == 0 && g == 0 && b == 0);
            norm_valid[i] = v;
            if v {
                nv += 1;
                for c in 0..3 {
                    let x = rgba[i * 4 + c] as f64;
                    sum[c] += x;
                    sq[c] += x * x;
                }
            }
        }
        let (mut mean, mut std) = ([0f32; 3], [1f32; 3]);
        if nv > 0 {
            for c in 0..3 {
                let m = sum[c] / nv as f64;
                let s = (sq[c] / nv as f64 - m * m).max(0.0).sqrt();
                mean[c] = m as f32;
                std[c] = if s > 1e-6 { s as f32 } else { 1.0 };
            }
        }

        // ── build padded (to /32) NCHW input ──
        let hp = h + (32 - h % 32) % 32;
        let wp = w + (32 - w % 32) % 32;
        let plane = hp * wp;
        let mut input = vec![0f32; 3 * plane];
        for i in 0..w * h {
            if !norm_valid[i] {
                continue; // black / out-of-footprint pixels fed as 0 (OCM nodata)
            }
            let (y, x) = (i / w, i % w);
            for c in 0..3 {
                input[c * plane + y * wp + x] = (rgba[i * 4 + c] as f32 - mean[c]) / std[c];
            }
        }

        // ── mean-logit ensemble at the original H×W ──
        const NCLS: usize = 4;
        let mut acc = vec![0f32; NCLS * w * h];
        for sess in self.sessions.iter_mut() {
            let tensor = Tensor::from_array((vec![1i64, 3, hp as i64, wp as i64], input.clone()))?;
            let outputs = sess.run(ort::inputs!["input" => tensor])?;
            let (_shape, data) = outputs["logits"].try_extract_tensor::<f32>()?; // [1,4,hp,wp]
            for cl in 0..NCLS {
                for y in 0..h {
                    for x in 0..w {
                        acc[cl * w * h + y * w + x] += data[cl * plane + y * wp + x];
                    }
                }
            }
        }
        let ns = self.sessions.len() as f32;

        // ── softmax + tuned labelling; validity = footprint (alpha > 0) ──
        let mut labels = vec![NODATA; w * h];
        for i in 0..w * h {
            if rgba[i * 4 + 3] == 0 {
                continue; // outside footprint = NODATA
            }
            let (y, x) = (i / w, i % w);
            let mut lg = [0f32; NCLS];
            for (cl, v) in lg.iter_mut().enumerate() {
                *v = acc[cl * w * h + y * w + x] / ns;
            }
            let mx = lg.iter().copied().fold(f32::MIN, f32::max);
            let mut e = [0f32; NCLS];
            let mut se = 0f32;
            for cl in 0..NCLS {
                e[cl] = (lg[cl] - mx).exp();
                se += e[cl];
            }
            let p_cloud = (e[1] + e[2]) / se; // thick + thin
            let argmax = (0..NCLS)
                .max_by(|a, b| lg[*a].partial_cmp(&lg[*b]).unwrap())
                .unwrap();
            labels[i] = if p_cloud >= self.tau {
                1 // cloud
            } else if argmax == 3 {
                2 // shadow
            } else {
                0 // clear
            };
        }
        Ok(labels)
    }
}

/// Decode a quicklook PNG (RGBA) and run the OCM ensemble → georeferenced
/// [`LabelRaster`]. `bbox` = `[min_lon, min_lat, max_lon, max_lat]` of the image
/// footprint (the quicklook spans this bbox, north-up). Ready to pass to
/// [`crate::cloud::fragment_cloud_fractions`].
pub fn label_raster_from_quicklook(
    model: &mut OcmEnsemble,
    png_path: &Path,
    bbox: [f64; 4],
) -> Result<LabelRaster, Box<dyn std::error::Error>> {
    let img = image::open(png_path)?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let labels = model.predict_labels(img.as_raw(), w, h)?;
    Ok(LabelRaster {
        width: w,
        height: h,
        bbox,
        labels,
    })
}
