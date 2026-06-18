use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::problem::SimsDiscreteProblem;

/// Errors that can occur during DZN serialisation / deserialisation.
#[derive(Debug, thiserror::Error)]
pub enum DznError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing field '{0}'")]
    MissingField(&'static str),
    #[error("parse error in field '{field}': {detail}")]
    ParseError { field: &'static str, detail: String },
}

// ── Write ─────────────────────────────────────────────────────────────────────

impl SimsDiscreteProblem {
    /// Serialise to the MiniZinc data (.dzn) format used by the PLS solver.
    ///
    /// Sets are written in **1-indexed** form (MiniZinc convention).
    pub fn to_dzn(&self, path: &Path) -> Result<(), DznError> {
        let mut out = String::new();

        writeln!(out, "num_images = {};", self.num_images).unwrap();
        writeln!(out, "universe = {};", self.universe).unwrap();
        writeln!(out, "images = {};", fmt_set_list(&self.images, 1)).unwrap();
        writeln!(out, "costs = {};", fmt_int_list(&self.costs)).unwrap();
        writeln!(out, "clouds = {};", fmt_set_list(&self.clouds, 1)).unwrap();
        writeln!(out, "areas = {};", fmt_int_list(&self.areas)).unwrap();
        writeln!(out, "resolution = {};", fmt_int_list(&self.resolution)).unwrap();
        writeln!(
            out,
            "incidence_angle = {};",
            fmt_int_list(&self.incidence_angle)
        )
        .unwrap();
        writeln!(out, "max_cloud_area = {};", self.max_cloud_area).unwrap();

        std::fs::write(path, out)?;
        Ok(())
    }
}

// ── Read ──────────────────────────────────────────────────────────────────────

impl SimsDiscreteProblem {
    /// Parse a MiniZinc .dzn file.
    ///
    /// Sets are expected in **1-indexed** form and are converted to **0-indexed**.
    pub fn from_dzn(path: &Path) -> Result<Self, DznError> {
        let content = std::fs::read_to_string(path)?;
        let mut fields = parse_fields(&content);

        macro_rules! take_scalar {
            ($name:literal, $ty:ty) => {
                fields
                    .remove($name)
                    .ok_or(DznError::MissingField($name))?
                    .trim()
                    .parse::<$ty>()
                    .map_err(|e| DznError::ParseError {
                        field: $name,
                        detail: e.to_string(),
                    })?
            };
        }

        let num_images = take_scalar!("num_images", usize);
        let universe = take_scalar!("universe", usize);
        let max_cloud_area = take_scalar!("max_cloud_area", i64);

        let costs = parse_int_list(
            fields
                .remove("costs")
                .ok_or(DznError::MissingField("costs"))?
                .trim(),
            "costs",
        )?;
        let areas = parse_int_list(
            fields
                .remove("areas")
                .ok_or(DznError::MissingField("areas"))?
                .trim(),
            "areas",
        )?;
        let resolution = parse_int_list(
            fields
                .remove("resolution")
                .ok_or(DznError::MissingField("resolution"))?
                .trim(),
            "resolution",
        )?;
        let incidence_angle = parse_int_list(
            fields
                .remove("incidence_angle")
                .ok_or(DznError::MissingField("incidence_angle"))?
                .trim(),
            "incidence_angle",
        )?;

        let images = parse_set_list(
            fields
                .remove("images")
                .ok_or(DznError::MissingField("images"))?
                .trim(),
            "images",
        )?;
        let clouds = parse_set_list(
            fields
                .remove("clouds")
                .ok_or(DznError::MissingField("clouds"))?
                .trim(),
            "clouds",
        )?;

        Ok(Self {
            num_images,
            universe,
            images,
            costs,
            clouds,
            areas,
            resolution,
            incidence_angle,
            max_cloud_area,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split `content` into field-name → raw-value pairs, stripping the trailing `;`.
fn parse_fields(content: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    // DZN values may span multiple lines; reconstruct them by joining at `;\n`.
    // Simple line-by-line split works because the Python writer emits one field per line.
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].trim().to_owned();
            let val = line[eq + 1..]
                .trim()
                .trim_end_matches(';')
                .trim()
                .to_owned();
            map.insert(key, val);
        }
    }
    map
}

fn parse_int_list(s: &str, field: &'static str) -> Result<Vec<i64>, DznError> {
    let inner = s.trim_start_matches('[').trim_end_matches(']');
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(|tok| {
            tok.trim().parse::<i64>().map_err(|e| DznError::ParseError {
                field,
                detail: e.to_string(),
            })
        })
        .collect()
}

/// Parse `[{1, 2}, {}, {3}]` and convert from 1-indexed to 0-indexed.
fn parse_set_list(s: &str, field: &'static str) -> Result<Vec<Vec<usize>>, DznError> {
    let inner = s.trim_start_matches('[').trim_end_matches(']').trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::new();
    // Walk through the string tracking brace depth to split on top-level commas
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in inner.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let set_str = inner[start..=i].trim();
                    result.push(parse_one_set(set_str, field)?);
                    start = i + 1;
                    // skip the following comma
                }
            }
            ',' if depth == 0 => {
                start = i + 1;
            }
            _ => {}
        }
    }

    Ok(result)
}

fn parse_one_set(s: &str, field: &'static str) -> Result<Vec<usize>, DznError> {
    let inner = s
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    let mut elems: Vec<usize> = inner
        .split(',')
        .map(|tok| {
            let v = tok
                .trim()
                .parse::<usize>()
                .map_err(|e| DznError::ParseError {
                    field,
                    detail: e.to_string(),
                })?;
            if v == 0 {
                return Err(DznError::ParseError {
                    field,
                    detail: "index 0 is invalid (DZN uses 1-based indexing)".into(),
                });
            }
            Ok(v - 1) // 1-based → 0-based
        })
        .collect::<Result<_, _>>()?;
    elems.sort_unstable();
    Ok(elems)
}

fn fmt_int_list(vals: &[i64]) -> String {
    let inner: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    format!("[{}]", inner.join(", "))
}

/// Format a list of index sets.  `offset` = 1 for 1-based DZN output.
fn fmt_set_list(sets: &[Vec<usize>], offset: usize) -> String {
    let parts: Vec<String> = sets
        .iter()
        .map(|s| {
            if s.is_empty() {
                "{}".to_owned()
            } else {
                let elems: Vec<String> = s.iter().map(|&i| (i + offset).to_string()).collect();
                format!("{{{}}}", elems.join(", "))
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::SimsDiscreteProblem;
    use tempfile::NamedTempFile;

    fn sample_problem() -> SimsDiscreteProblem {
        SimsDiscreteProblem {
            num_images: 2,
            universe: 3,
            images: vec![vec![0, 1], vec![1, 2]],
            costs: vec![100, 200],
            clouds: vec![vec![0], vec![]],
            areas: vec![10000, 20000, 15000],
            resolution: vec![60, 80],
            incidence_angle: vec![150, 200],
            max_cloud_area: 45000,
        }
    }

    #[test]
    fn test_roundtrip_dzn() {
        let original = sample_problem();
        let tmp = NamedTempFile::new().unwrap();
        original.to_dzn(tmp.path()).unwrap();

        let loaded = SimsDiscreteProblem::from_dzn(tmp.path()).unwrap();
        assert_eq!(loaded.num_images, original.num_images);
        assert_eq!(loaded.universe, original.universe);
        assert_eq!(loaded.images, original.images);
        assert_eq!(loaded.costs, original.costs);
        assert_eq!(loaded.clouds, original.clouds);
        assert_eq!(loaded.areas, original.areas);
        assert_eq!(loaded.resolution, original.resolution);
        assert_eq!(loaded.incidence_angle, original.incidence_angle);
        assert_eq!(loaded.max_cloud_area, original.max_cloud_area);
    }

    #[test]
    fn test_fmt_set_list_1indexed() {
        let sets = vec![vec![0usize, 1], vec![], vec![2usize]];
        let s = fmt_set_list(&sets, 1);
        assert_eq!(s, "[{1, 2}, {}, {3}]");
    }
}
