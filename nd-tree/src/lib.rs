#![feature(adt_const_params)]

pub mod nd_tree;
pub mod tracked_nd_tree;

// Re-export key types for easier access
pub use nd_tree::{NDTree, ScalarizedQueryResult, ScalarizedQueryStats};
pub use tracked_nd_tree::{InsertionResult, TrackedNdTree, TrackedNdTreeConfig};
