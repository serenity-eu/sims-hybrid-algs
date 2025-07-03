use std::time::{Duration, Instant};

pub struct Timer {
    start: Instant,
    duration: Duration,
}

impl Timer {
    #[must_use]
    pub fn start(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
        }
    }

    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }

    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.start.elapsed() > self.duration
    }
}
