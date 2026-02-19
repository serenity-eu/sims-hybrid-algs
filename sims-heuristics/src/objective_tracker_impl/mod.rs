pub mod standard_trackers;
pub mod tracker_trace;
pub mod simd_trackers;
pub mod proven_safe_trackers;

// Core trackers (always compiled)
mod composite_debug_trackers;

// Additional tracker implementations (compiled only with "additional_trackers" feature)
#[cfg(feature = "additional_trackers")]
pub mod alternative_trackers;
#[cfg(feature = "additional_trackers")]
pub mod explicit_simd_trackers;
#[cfg(feature = "additional_trackers")]
pub mod safe_simd_trackers;
#[cfg(feature = "additional_trackers")]
pub mod saturating_trackers;
#[cfg(feature = "additional_trackers")]
pub mod segment_tree_trackers;
#[cfg(feature = "additional_trackers")]
pub mod simplified_trackers;
