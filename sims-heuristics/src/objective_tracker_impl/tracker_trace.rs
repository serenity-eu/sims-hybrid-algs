//! Tracker operation tracing for benchmarks and validation.
//!
//! This module provides infrastructure to record tracker operations to a binary trace file.
//! The trace can later be replayed to validate that different tracker implementations
//! produce identical results.
//!
//! # Trace Format
//!
//! Each record is a little-endian u16: `(op << 12) | image_index`
//!
//! Op codes:
//! - 0: TrackAdd - track_image_addition called
//! - 1: TrackRem - track_image_removal called  
//! - 2: PeekAdd - peek_addition_delta called
//! - 3: PeekRem - peek_removal_delta called
//! - 4: Reset - new() or initialize_from() called
//!
//! # Usage
//!
//! Enable tracing at compile time with `--features trace-tracker-ops`, then:
//!
//! ```ignore
//! use pls::objective_tracker_impl::tracker_trace;
//!
//! // Initialize the trace writer (once at program start)
//! tracker_trace::init("my_trace.u16").unwrap();
//!
//! // ... run your algorithm using StandardTrackerArray ...
//!
//! // Flush and close the trace file
//! tracker_trace::finish();
//! ```

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Operation codes for trace events.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TraceOp {
    TrackAdd = 0,
    TrackRem = 1,
    PeekAdd = 2,
    PeekRem = 3,
    Reset = 4,
}

// Thread-local trace writer
thread_local! {
    static TRACE_WRITER: RefCell<Option<BufWriter<File>>> = const { RefCell::new(None) };
}

/// Initialize trace recording to the specified file path.
///
/// Returns an error if the file cannot be created.
///
/// # Errors
///
/// Returns an IO error if the file cannot be created.
pub fn init<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer
    TRACE_WRITER.with_borrow_mut(|w| *w = Some(writer));
    Ok(())
}

/// Check if trace recording is active.
#[inline]
#[must_use]
pub fn is_active() -> bool {
    TRACE_WRITER.with_borrow(|w| w.is_some())
}

/// Record a trace event.
///
/// This is a no-op if tracing is not initialized.
#[inline]
pub fn record(op: TraceOp, image_index: usize) {
    TRACE_WRITER.with_borrow_mut(|writer| {
        if let Some(w) = writer.as_mut() {
            // Encode: (op << 12) | (image_index & 0x0FFF)
            // This supports up to 4096 images per record
            #[allow(clippy::cast_possible_truncation)]
            let record = ((op as u16) << 12) | (image_index as u16 & 0x0FFF);
            let _ = w.write_all(&record.to_le_bytes());
        }
    });
}

/// Flush and close the trace file.
///
/// Call this when tracing is complete to ensure all data is written.
pub fn finish() {
    TRACE_WRITER.with_borrow_mut(|writer| {
        if let Some(w) = writer.as_mut() {
            let _ = w.flush();
        }
        *writer = None;
    });
}

/// Get statistics about the current trace session.
#[must_use]
pub fn stats() -> Option<TraceStats> {
    TRACE_WRITER.with_borrow(|writer| {
        writer.as_ref().map(|w| TraceStats {
            buffer_capacity: w.capacity(),
        })
    })
}

/// Statistics about the trace session.
#[derive(Debug, Clone)]
pub struct TraceStats {
    /// Capacity of the write buffer in bytes.
    pub buffer_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_trace_encoding() {
        let trace_path = std::env::temp_dir().join("tracker_trace_test.u16");

        // Initialize tracing
        init(&trace_path).unwrap();
        assert!(is_active());

        // Record some events
        record(TraceOp::Reset, 0);
        record(TraceOp::TrackAdd, 42);
        record(TraceOp::PeekRem, 100);
        record(TraceOp::TrackRem, 255);
        record(TraceOp::PeekAdd, 4095); // Max index

        finish();
        assert!(!is_active());

        // Read and verify
        let mut file = File::open(&trace_path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();

        assert_eq!(bytes.len(), 10); // 5 records * 2 bytes each

        // Decode records
        let records: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(records[0], (4 << 12) | 0); // Reset, index 0
        assert_eq!(records[1], (0 << 12) | 42); // TrackAdd, index 42
        assert_eq!(records[2], (3 << 12) | 100); // PeekRem, index 100
        assert_eq!(records[3], (1 << 12) | 255); // TrackRem, index 255
        assert_eq!(records[4], (2 << 12) | 4095); // PeekAdd, index 4095

        // Cleanup
        let _ = std::fs::remove_file(&trace_path);
    }
}
