//! # Solver Selection Module
//!
//! This module provides an enumeration of available LP/MILP solvers
//! that can be used with the AUGMECON implementation.

/// Available LP/MILP solvers for AUGMECON optimization
///
/// This enum provides access to common solvers supported by `good_lp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Solver {
    /// Default solver provided by `good_lp` (usually CBC)
    Default,
    /// COIN-OR CBC solver - open source MILP solver
    CoinCbc,
    /// `HiGHS` solver - high performance linear programming solver
    HiGHS,
}

impl Solver {
    
    /// Get a human-readable name for the solver
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::CoinCbc => "COIN-OR CBC",
            Self::HiGHS => "HiGHS",
        }
    }
    
    /// Check if this solver supports setting custom parameters
    #[must_use]
    pub const fn supports_parameters(self) -> bool {
        match self {
            Self::CoinCbc => true,
            Self::Default | Self::HiGHS => false, // HiGHS doesn't support generic parameters
        }
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::Default
    }
}

impl std::fmt::Display for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for Solver {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "coin_cbc" | "cbc" | "coin-cbc" => Ok(Self::CoinCbc),
            "highs" | "hi_ghs" => Ok(Self::HiGHS),
            _ => Err(format!("Unknown solver: {s}. Available solvers: default, coin_cbc, highs")),
        }
    }
}
