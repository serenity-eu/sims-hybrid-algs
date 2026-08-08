//! Turn a georeferenced cloud **label raster** (the U-Net output for one image's
//! quicklook) into a per-fragment cloud+shadow fraction for the SIMS problem.
//!
//! The cloud-mask model (`cloud_unet.h5`) emits a 3-class label per pixel.
//! The channel→class mapping was confirmed empirically (running the net on our
//! quicklooks): `0 = no-cloud/clear`, `1 = cloud`, `2 = cloud-shadow` — note
//! this is the *reverse* of the README's prose ordering. For satellite-image
//! mosaic selection both cloud and shadow are unusable, so [`is_cloudy`] treats
//! classes `{1, 2}` as cloudy. Nodata pixels (outside the footprint, from the
//! quicklook's alpha channel) carry [`NODATA`] and are excluded.
//!
//! A quicklook is a north-up render spanning the image footprint's geographic
//! bounding box, so pixel↔lon/lat is a simple affine ([`LabelRaster::pixel_center`]).
//! This module has **no** image-decode or model dependency — the caller supplies
//! the decoded, georeferenced labels; inference and PNG decoding live elsewhere.

use geo::{BoundingRect, LineString, MultiPolygon, Polygon, Simplify};

/// Sentinel label for pixels with no valid data (footprint alpha == 0).
pub const NODATA: u8 = 255;

/// SIMS "cloudy" = cloud (1) OR cloud-shadow (2); no-cloud (0) and [`NODATA`] are not.
#[inline]
pub fn is_cloudy(label: u8) -> bool {
    label == 1 || label == 2
}

/// A north-up, geo-referenced label raster for a single image's footprint.
///
/// `labels` is row-major, `width * height`; row 0 is the **north** edge (`maxy`).
/// Each entry is a class label (`0/1/2`) or [`NODATA`].
#[derive(Debug, Clone)]
pub struct LabelRaster {
    pub width: usize,
    pub height: usize,
    /// Geographic bounds `[min_lon, min_lat, max_lon, max_lat]` of the footprint.
    pub bbox: [f64; 4],
    pub labels: Vec<u8>,
}

impl LabelRaster {
    /// Lon/lat of the centre of pixel `(col, row)`.
    #[inline]
    pub fn pixel_center(&self, col: usize, row: usize) -> (f64, f64) {
        let [min_x, min_y, max_x, max_y] = self.bbox;
        let lon = min_x + (col as f64 + 0.5) / self.width as f64 * (max_x - min_x);
        let lat = max_y - (row as f64 + 0.5) / self.height as f64 * (max_y - min_y);
        (lon, lat)
    }

    #[inline]
    fn label(&self, col: usize, row: usize) -> u8 {
        self.labels[row * self.width + col]
    }

    /// Inclusive pixel-column/row window covering a geographic bbox, clamped to
    /// the raster. Returns `None` if the bbox lies fully outside.
    fn pixel_window(
        &self,
        gx0: f64,
        gy0: f64,
        gx1: f64,
        gy1: f64,
    ) -> Option<(usize, usize, usize, usize)> {
        let [min_x, min_y, max_x, max_y] = self.bbox;
        let (w, h) = (self.width as f64, self.height as f64);
        // col grows with lon; row grows as lat decreases (north-up).
        let c0 = ((gx0 - min_x) / (max_x - min_x) * w).floor();
        let c1 = ((gx1 - min_x) / (max_x - min_x) * w).ceil();
        let r0 = ((max_y - gy1) / (max_y - min_y) * h).floor();
        let r1 = ((max_y - gy0) / (max_y - min_y) * h).ceil();
        if c1 < 0.0 || r1 < 0.0 || c0 >= w || r0 >= h {
            return None;
        }
        let c0 = c0.max(0.0) as usize;
        let r0 = r0.max(0.0) as usize;
        let c1 = (c1.min(w) as usize).max(1) - 1; // clamp to last index
        let r1 = (r1.min(h) as usize).max(1) - 1;
        Some((c0, r0, c1, r1))
    }
}

/// Even-odd ray-cast point-in-ring test.
fn point_in_ring(ring: &LineString<f64>, x: f64, y: f64) -> bool {
    let pts = &ring.0;
    if pts.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let (xi, yi) = (pts[i].x, pts[i].y);
        let (xj, yj) = (pts[j].x, pts[j].y);
        if (yi > y) != (yj > y) {
            let xc = xi + (y - yi) / (yj - yi) * (xj - xi);
            if x < xc {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[inline]
fn point_in_polygon(poly: &Polygon<f64>, x: f64, y: f64) -> bool {
    point_in_ring(poly.exterior(), x, y) && !poly.interiors().iter().any(|h| point_in_ring(h, x, y))
}

/// For each fragment polygon, the fraction of its **valid** pixels (excluding
/// [`NODATA`]) that are cloudy (cloud+shadow), sampled at pixel centres of
/// `raster`. Fragments with no valid pixels yield `0.0`.
///
/// Each fragment must lie within the raster's footprint (which holds by
/// construction: a fragment assigned to an image is inside that image's
/// footprint). Only the fragment's pixel bounding box is scanned, so total work
/// is ~O(number of pixels in the footprint), not O(fragments × pixels).
pub fn fragment_cloud_fractions(raster: &LabelRaster, fragments: &[Polygon<f64>]) -> Vec<f64> {
    fragments
        .iter()
        .map(|frag| {
            let Some(rect) = frag.bounding_rect() else {
                return 0.0;
            };
            let win = raster.pixel_window(rect.min().x, rect.min().y, rect.max().x, rect.max().y);
            let Some((c0, r0, c1, r1)) = win else {
                return 0.0;
            };
            let (mut valid, mut cloudy) = (0usize, 0usize);
            for row in r0..=r1 {
                for col in c0..=c1 {
                    let lab = raster.label(col, row);
                    if lab == NODATA {
                        continue;
                    }
                    let (lon, lat) = raster.pixel_center(col, row);
                    if point_in_polygon(frag, lon, lat) {
                        valid += 1;
                        if is_cloudy(lab) {
                            cloudy += 1;
                        }
                    }
                }
            }
            if valid == 0 {
                0.0
            } else {
                cloudy as f64 / valid as f64
            }
        })
        .collect()
}

/// Binarize per-fragment fractions into the indices of fragments that count as
/// cloudy for the SIMS `clouds[image]` set (fraction ≥ `threshold`, e.g. 0.5).
pub fn cloudy_fragments(fractions: &[f64], threshold: f64) -> Vec<usize> {
    fractions
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| (f >= threshold).then_some(i))
        .collect()
}

/// Vectorize the selected (cloud) regions of a georeferenced label raster into a
/// [`MultiPolygon`] in lon/lat: **each connected cloud region becomes one
/// polygon** (with holes for internal gaps), and a polygon's exterior ring is
/// that cloud's border. `select` chooses which labels count as cloud — e.g.
/// `|l| l == 1` for cloud only, or [`is_cloudy`] for cloud+shadow.
///
/// Implemented with marching squares (`contour`) at threshold 0.5 over the
/// binary cloud/not-cloud field, using the raster's bbox affine so the output
/// coordinates are geographic. `smooth = false` keeps borders on the pixel grid.
pub fn cloud_multipolygon(raster: &LabelRaster, select: impl Fn(u8) -> bool) -> MultiPolygon<f64> {
    let (w, h) = (raster.width, raster.height);
    if w == 0 || h == 0 {
        return MultiPolygon::new(vec![]);
    }
    let [min_x, min_y, max_x, max_y] = raster.bbox;
    // Feed rows bottom-up (row 0 = south) with a *positive* y_step: contour
    // classifies exterior vs hole by ring-area sign, so a negative step would
    // flip winding and drop every exterior. This keeps output north-up lon/lat.
    let mut values = vec![0.0f64; w * h];
    for row in 0..h {
        let (src, dst) = (row * w, (h - 1 - row) * w);
        for col in 0..w {
            if select(raster.labels[src + col]) {
                values[dst + col] = 1.0;
            }
        }
    }
    let builder = contour::ContourBuilder::new(w, h, false)
        .x_origin(min_x)
        .y_origin(min_y)
        .x_step((max_x - min_x) / w as f64)
        .y_step((max_y - min_y) / h as f64);
    match builder.contours(&values, &[0.5]) {
        Ok(mut cs) if !cs.is_empty() => cs.swap_remove(0).into_inner().0,
        _ => MultiPolygon::new(vec![]),
    }
}

/// Simplify cloud polygons to cut vertex count and remove the marching-squares
/// pixel-staircase, via Ramer–Douglas–Peucker. `tolerance` is the epsilon in the
/// raster's coordinate (degree) units; a good value is ~1–2 pixels, i.e.
/// `k * (max_lon - min_lon) / width`. RDP preserves corners and endpoints and
/// bounds the deviation (and hence area change) by `tolerance` — no rounding,
/// no corner-cutting. `tolerance <= 0` returns a clone unchanged.
pub fn simplify_polygons(mp: &MultiPolygon<f64>, tolerance: f64) -> MultiPolygon<f64> {
    if tolerance > 0.0 {
        mp.simplify(tolerance)
    } else {
        mp.clone()
    }
}

/// Total vertex count of a MultiPolygon (all exterior + interior ring points).
pub fn vertex_count(mp: &MultiPolygon<f64>) -> usize {
    mp.0.iter()
        .map(|p| p.exterior().0.len() + p.interiors().iter().map(|r| r.0.len()).sum::<usize>())
        .sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Area, Centroid, polygon};

    /// A 2×2 cloud block inside a 4×4 grid over bbox [0,0]–[4,4] vectorizes to
    /// exactly one polygon of area ~4, centred on the block.
    #[test]
    fn test_cloud_multipolygon_single_blob() {
        let mut labels = vec![0u8; 16];
        for row in 1..=2 {
            for col in 1..=2 {
                labels[row * 4 + col] = 1; // cloud block, rows 1-2 cols 1-2
            }
        }
        let raster = LabelRaster {
            width: 4,
            height: 4,
            bbox: [0.0, 0.0, 4.0, 4.0],
            labels,
        };
        let mp = cloud_multipolygon(&raster, |l| l == 1);
        assert_eq!(mp.0.len(), 1, "one connected cloud -> one polygon");
        let area = mp.unsigned_area();
        assert!((3.5..=4.5).contains(&area), "area ~4, got {area}");
        let c = mp.centroid().unwrap();
        assert!(
            (c.x() - 2.0).abs() < 0.3 && (c.y() - 2.0).abs() < 0.3,
            "centroid ~(2,2), got {c:?}"
        );
    }

    /// No cloud -> empty MultiPolygon.
    #[test]
    fn test_cloud_multipolygon_empty() {
        let raster = LabelRaster {
            width: 3,
            height: 3,
            bbox: [0.0, 0.0, 3.0, 3.0],
            labels: vec![0; 9],
        };
        assert!(cloud_multipolygon(&raster, |l| l == 1).0.is_empty());
    }

    /// A staircase (diagonal) cloud border has many pixel-step points; RDP
    /// simplification collapses them without dropping the polygon.
    #[test]
    fn test_simplify_reduces_points() {
        let (w, h) = (8usize, 8usize);
        let mut labels = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..=row {
                labels[row * w + col] = 1; // lower-left staircase triangle
            }
        }
        let raster = LabelRaster {
            width: w,
            height: h,
            bbox: [0.0, 0.0, 8.0, 8.0],
            labels,
        };
        let raw = cloud_multipolygon(&raster, |l| l == 1);
        assert!(!raw.0.is_empty());
        let n_raw = vertex_count(&raw);
        let simplified = simplify_polygons(&raw, 1.5);
        assert!(!simplified.0.is_empty(), "polygon survives simplification");
        assert!(
            vertex_count(&simplified) < n_raw,
            "simplify should cut points: {n_raw} -> {}",
            vertex_count(&simplified)
        );
    }

    /// Footprint [0,0]–[2,1], 4×2 label grid: left two columns cloudy, right two
    /// clear. Left fragment must read 100% cloudy, right 0%.
    #[test]
    fn test_left_half_cloudy() {
        let raster = LabelRaster {
            width: 4,
            height: 2,
            bbox: [0.0, 0.0, 2.0, 1.0],
            // row-major, row0 = north. cols 0,1 -> cloud(1); cols 2,3 -> clear(0)
            labels: vec![1, 1, 0, 0, /* row0 */ 1, 1, 0, 0 /* row1 */],
        };
        let left = polygon![(x:0.0,y:0.0),(x:1.0,y:0.0),(x:1.0,y:1.0),(x:0.0,y:1.0)];
        let right = polygon![(x:1.0,y:0.0),(x:2.0,y:0.0),(x:2.0,y:1.0),(x:1.0,y:1.0)];

        let fr = fragment_cloud_fractions(&raster, &[left, right]);
        assert!(
            (fr[0] - 1.0).abs() < 1e-9,
            "left fully cloudy, got {}",
            fr[0]
        );
        assert!(fr[1].abs() < 1e-9, "right clear, got {}", fr[1]);
        assert_eq!(cloudy_fragments(&fr, 0.5), vec![0]);
    }

    /// Shadow (class 2) counts as cloudy; nodata (255) is excluded from the
    /// denominator.
    #[test]
    fn test_shadow_counts_and_nodata_excluded() {
        let raster = LabelRaster {
            width: 2,
            height: 1,
            bbox: [0.0, 0.0, 2.0, 1.0],
            labels: vec![2 /*shadow*/, NODATA],
        };
        let full = polygon![(x:0.0,y:0.0),(x:2.0,y:0.0),(x:2.0,y:1.0),(x:0.0,y:1.0)];
        let fr = fragment_cloud_fractions(&raster, &[full]);
        // one valid pixel (shadow) + one nodata -> 1/1 cloudy.
        assert!(
            (fr[0] - 1.0).abs() < 1e-9,
            "shadow should be cloudy, got {}",
            fr[0]
        );
    }

    #[test]
    fn test_pixel_center_affine() {
        let r = LabelRaster {
            width: 4,
            height: 2,
            bbox: [0.0, 0.0, 2.0, 1.0],
            labels: vec![0; 8],
        };
        let (lon, lat) = r.pixel_center(0, 0); // top-left pixel centre
        assert!(
            (lon - 0.25).abs() < 1e-9 && (lat - 0.75).abs() < 1e-9,
            "{lon},{lat}"
        );
    }
}
