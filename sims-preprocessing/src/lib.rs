pub mod cloud;
/// OmniCloudMask cloud-mask inference via ONNX Runtime. Enable with the `ocm`
/// feature; produces a [`cloud::LabelRaster`] for [`cloud::fragment_cloud_fractions`].
#[cfg(feature = "ocm")]
pub mod cloud_ocm;
pub mod clouds;
pub mod dzn;
pub mod fragmentize;
/// Reference/cross-check backend built on system libgeos. Enable with the
/// `geos` feature; the default [`fragmentize`] is pure Rust.
#[cfg(feature = "geos")]
pub mod fragmentize_geos;
pub mod problem;
pub mod sampler;

pub use fragmentize::fragmentize;

pub use problem::{SimsDiscreteProblem, SimsProblem};
