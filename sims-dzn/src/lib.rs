//! Fast single-pass parser for the SIMS `MiniZinc` `.dzn` instance format.
//!
//! The format is a fixed set of `name = value;` assignments: plain integers
//! (`num_images`, `universe`, `max_cloud_area`), integer arrays (`costs`,
//! `areas`, `resolution`, `incidence_angle`), and arrays of 1-based integer
//! sets (`images`, `clouds`).
//!
//! This parses the file as raw bytes (the format is pure ASCII, so no UTF-8
//! validation pass is needed) in a single forward scan, with hand-rolled
//! integer parsing — no regex, no per-field re-scan of the whole file. A
//! previous regex-based implementation re-scanned the *entire* file content
//! once per field (9 times total), which on a 100+ MB instance meant fields
//! near the end of the file (like `max_cloud_area`) paid the cost of
//! scanning past every byte of the huge `images`/`clouds` arrays multiple
//! times over.

use std::path::Path;

/// Raw, unprocessed contents of a `.dzn` SIMS instance file.
///
/// Set members in `images`/`clouds` are already converted from the file's
/// 1-based indexing to 0-based.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawSimsData {
    /// Number of images available.
    pub num_images: usize,
    /// Number of universe fragments to cover.
    pub universe: usize,
    /// For each image: the fragments it covers (0-based).
    pub images: Vec<Vec<usize>>,
    /// Cost of each image.
    pub costs: Vec<i64>,
    /// For each image: the fragments that are cloudy in it (0-based).
    pub clouds: Vec<Vec<usize>>,
    /// Area of each universe fragment.
    pub areas: Vec<i64>,
    /// Resolution of each image.
    pub resolution: Vec<i64>,
    /// Incidence angle of each image.
    pub incidence_angle: Vec<i64>,
    /// Maximum cloud area threshold.
    pub max_cloud_area: i64,
}

/// Error parsing a `.dzn` file.
#[derive(Debug, thiserror::Error)]
pub enum DznParseError {
    /// Failed to read the file from disk.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Reached end of input mid-token.
    #[error("unexpected end of input while parsing {context} (byte {pos})")]
    UnexpectedEof {
        /// What was being parsed when input ran out.
        context: &'static str,
        /// Byte offset where parsing stopped.
        pos: usize,
    },
    /// A specific byte was expected but not found.
    #[error("expected '{expected}' at byte {pos}, found '{found}'")]
    Expected {
        /// The byte that was required.
        expected: char,
        /// The byte actually found (or '\\0' at EOF).
        found: char,
        /// Byte offset of the mismatch.
        pos: usize,
    },
    /// A set element was <= 0 (the format is 1-based; 0 is invalid).
    #[error("invalid set element {value} at byte {pos} (must be >= 1)")]
    InvalidSetElement {
        /// The offending value.
        value: i64,
        /// Byte offset of the value.
        pos: usize,
    },
    /// A required top-level field was never assigned.
    #[error("missing required field: {0}")]
    MissingField(&'static str),
}

/// Parse a `.dzn` file from disk.
///
/// # Errors
/// Returns [`DznParseError`] if the file can't be read or doesn't match the
/// expected SIMS `.dzn` grammar.
pub fn parse_dzn_file(path: &Path) -> Result<RawSimsData, DznParseError> {
    let bytes = std::fs::read(path).map_err(|source| DznParseError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_dzn_bytes(&bytes)
}

/// Parse `.dzn` file contents already held in memory.
///
/// # Errors
/// Returns [`DznParseError`] if the content doesn't match the expected SIMS
/// `.dzn` grammar.
pub fn parse_dzn_bytes(bytes: &[u8]) -> Result<RawSimsData, DznParseError> {
    let mut parser = Parser { bytes, pos: 0 };
    let mut data = RawSimsData::default();
    let mut have_num_images = false;
    let mut have_universe = false;
    let mut have_max_cloud_area = false;

    while !parser.at_end() {
        let name = parser.read_identifier()?;
        parser.expect(b'=')?;
        match name {
            "num_images" => {
                data.num_images = parser.parse_uint()?;
                have_num_images = true;
            }
            "universe" => {
                data.universe = parser.parse_uint()?;
                have_universe = true;
            }
            "max_cloud_area" => {
                data.max_cloud_area = parser.parse_int()?;
                have_max_cloud_area = true;
            }
            "costs" => data.costs = parser.parse_int_array()?,
            "areas" => data.areas = parser.parse_int_array()?,
            "resolution" => data.resolution = parser.parse_int_array()?,
            "incidence_angle" => data.incidence_angle = parser.parse_int_array()?,
            "images" => data.images = parser.parse_set_array()?,
            "clouds" => data.clouds = parser.parse_set_array()?,
            _ => parser.skip_value()?,
        }
        parser.expect(b';')?;
    }

    if !have_num_images {
        return Err(DznParseError::MissingField("num_images"));
    }
    if !have_universe {
        return Err(DznParseError::MissingField("universe"));
    }
    if !have_max_cloud_area {
        return Err(DznParseError::MissingField("max_cloud_area"));
    }

    Ok(data)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws_and_comments(&mut self) {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.bytes.get(self.pos) == Some(&b'%') {
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_ws_and_comments();
        self.pos >= self.bytes.len()
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws_and_comments();
        self.bytes.get(self.pos).copied()
    }

    fn read_identifier(&mut self) -> Result<&'a str, DznParseError> {
        self.skip_ws_and_comments();
        let start = self.pos;
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(DznParseError::UnexpectedEof {
                context: "identifier",
                pos: self.pos,
            });
        }
        // Only ASCII alphanumeric/underscore bytes were consumed above, so
        // this slice is always valid UTF-8.
        Ok(str::from_utf8(&self.bytes[start..self.pos]).unwrap_or_default())
    }

    fn expect(&mut self, expected: u8) -> Result<(), DznParseError> {
        self.skip_ws_and_comments();
        if self.bytes.get(self.pos) != Some(&expected) {
            let found = self.bytes.get(self.pos).map_or('\0', |&b| b as char);
            return Err(DznParseError::Expected {
                expected: expected as char,
                found,
                pos: self.pos,
            });
        }
        self.pos += 1;
        Ok(())
    }

    /// Parses a (possibly signed) decimal integer with no allocation.
    fn parse_int(&mut self) -> Result<i64, DznParseError> {
        self.skip_ws_and_comments();
        let negative = self.bytes.get(self.pos) == Some(&b'-');
        if negative {
            self.pos += 1;
        }
        let start = self.pos;
        let mut value: i64 = 0;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() {
                value = value * 10 + i64::from(b - b'0');
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(DznParseError::UnexpectedEof {
                context: "integer",
                pos: self.pos,
            });
        }
        Ok(if negative { -value } else { value })
    }

    fn parse_uint(&mut self) -> Result<usize, DznParseError> {
        let pos = self.pos;
        let value = self.parse_int()?;
        usize::try_from(value).map_err(|_| DznParseError::InvalidSetElement { value, pos })
    }

    fn parse_int_array(&mut self) -> Result<Vec<i64>, DznParseError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                Some(b',') => self.pos += 1,
                Some(_) => out.push(self.parse_int()?),
                None => {
                    return Err(DznParseError::UnexpectedEof {
                        context: "int array",
                        pos: self.pos,
                    })
                }
            }
        }
        Ok(out)
    }

    /// Parses a `{1, 2, 3}`-style set, converting 1-based elements to 0-based.
    fn parse_uint_set(&mut self) -> Result<Vec<usize>, DznParseError> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                Some(b',') => self.pos += 1,
                Some(_) => {
                    let pos = self.pos;
                    let value = self.parse_int()?;
                    let value = usize::try_from(value)
                        .ok()
                        .filter(|&v| v >= 1)
                        .ok_or(DznParseError::InvalidSetElement { value, pos })?;
                    out.push(value - 1);
                }
                None => {
                    return Err(DznParseError::UnexpectedEof {
                        context: "set",
                        pos: self.pos,
                    })
                }
            }
        }
        Ok(out)
    }

    fn parse_set_array(&mut self) -> Result<Vec<Vec<usize>>, DznParseError> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                Some(b',') => self.pos += 1,
                Some(b'{') => out.push(self.parse_uint_set()?),
                _ => {
                    return Err(DznParseError::UnexpectedEof {
                        context: "set array",
                        pos: self.pos,
                    })
                }
            }
        }
        Ok(out)
    }

    /// Skips an unrecognized field's value, up to (not including) its `;`.
    /// Tracks bracket/brace nesting so semicolons can't appear inside a
    /// value in this grammar, but stays defensive about it anyway.
    fn skip_value(&mut self) -> Result<(), DznParseError> {
        let mut depth: i32 = 0;
        loop {
            match self.bytes.get(self.pos) {
                Some(b'[' | b'{') => {
                    depth += 1;
                    self.pos += 1;
                }
                Some(b']' | b'}') => {
                    depth -= 1;
                    self.pos += 1;
                }
                Some(b';') if depth <= 0 => break,
                Some(_) => self.pos += 1,
                None => {
                    return Err(DznParseError::UnexpectedEof {
                        context: "value",
                        pos: self.pos,
                    })
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_instance() {
        let src = b"num_images = 2;\nuniverse = 3;\nimages = [{1,2},{2,3}];\ncosts = [10, 20];\nclouds = [{1},{}];\nareas = [5, 6, 7];\nresolution = [1, 2];\nincidence_angle = [10, 20];\nmax_cloud_area = 100;\n";
        let data = parse_dzn_bytes(src).unwrap();
        assert_eq!(data.num_images, 2);
        assert_eq!(data.universe, 3);
        assert_eq!(data.images, vec![vec![0, 1], vec![1, 2]]);
        assert_eq!(data.costs, vec![10, 20]);
        assert_eq!(data.clouds, vec![vec![0], vec![]]);
        assert_eq!(data.areas, vec![5, 6, 7]);
        assert_eq!(data.resolution, vec![1, 2]);
        assert_eq!(data.incidence_angle, vec![10, 20]);
        assert_eq!(data.max_cloud_area, 100);
    }

    #[test]
    fn field_order_is_flexible() {
        let src = b"max_cloud_area = 100;\nuniverse = 1;\nnum_images = 1;\nimages = [{1}];\nclouds = [{}];\ncosts = [1];\nareas = [1];\nresolution = [1];\nincidence_angle = [1];\n";
        let data = parse_dzn_bytes(src).unwrap();
        assert_eq!(data.num_images, 1);
        assert_eq!(data.max_cloud_area, 100);
    }

    #[test]
    fn unknown_fields_are_skipped() {
        let src = b"num_images = 1;\nuniverse = 1;\nsome_future_field = [1, 2, {3, 4}];\nimages = [{1}];\nclouds = [{}];\ncosts = [1];\nareas = [1];\nresolution = [1];\nincidence_angle = [1];\nmax_cloud_area = 1;\n";
        let data = parse_dzn_bytes(src).unwrap();
        assert_eq!(data.num_images, 1);
    }

    #[test]
    fn empty_set_parses_to_empty_vec() {
        let src = b"num_images = 1;\nuniverse = 1;\nimages = [{1}];\nclouds = [{}];\ncosts = [1];\nareas = [1];\nresolution = [1];\nincidence_angle = [1];\nmax_cloud_area = 1;\n";
        let data = parse_dzn_bytes(src).unwrap();
        assert_eq!(data.clouds, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn zero_based_set_element_is_rejected() {
        let src = b"num_images = 1;\nuniverse = 1;\nimages = [{0}];\nclouds = [{}];\ncosts = [1];\nareas = [1];\nresolution = [1];\nincidence_angle = [1];\nmax_cloud_area = 1;\n";
        assert!(matches!(
            parse_dzn_bytes(src),
            Err(DznParseError::InvalidSetElement { value: 0, .. })
        ));
    }

    #[test]
    fn missing_field_is_reported() {
        let src = b"num_images = 1;\nuniverse = 1;\n";
        assert!(matches!(
            parse_dzn_bytes(src),
            Err(DznParseError::MissingField("max_cloud_area"))
        ));
    }
}
