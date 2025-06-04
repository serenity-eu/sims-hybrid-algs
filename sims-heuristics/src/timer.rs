use std::time::{Duration, Instant};

pub struct Timer {
    start: Instant,
    duration: Duration,
}

impl Timer {
    pub fn start(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
        }
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn is_expired(&self) -> bool {
        self.start.elapsed() > self.duration
    }
}
