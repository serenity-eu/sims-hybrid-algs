//! Timer for tracking timeouts and elapsed time

use std::time::{Duration, Instant};

/// Timer for tracking elapsed time and checking timeouts
pub struct Timer {
    start: Instant,
    duration: Duration,
}

impl Timer {
    /// Start a new timer with the given duration
    #[must_use]
    pub fn start(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
        }
    }

    /// Get the total duration for this timer
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    /// Get the elapsed time since the timer started
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get the remaining time (returns `Duration::ZERO` if expired)
    #[must_use]
    pub fn remaining(&self) -> Duration {
        let elapsed = self.start.elapsed();
        if elapsed >= self.duration {
            Duration::ZERO
        } else {
            self.duration - elapsed
        }
    }

    /// Check if the timer has expired
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.start.elapsed() >= self.duration
    }
}
