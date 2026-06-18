use geo::{BoundingRect, Geometry as GeoGeometry, Polygon, Rect};
use geos::{Geom, Geometry as GeosGeometry};
use rstar::{RTree, RTreeObject, AABB};
use std::convert::TryFrom;

// ── Spatial index entry ───────────────────────────────────────────────────────

/// Wrapper that lets fragments live in an R-tree indexed by their bounding box.
struct FragmentEntry {
    idx: usize,
    envelope: AABB<[f64; 2]>,
}

impl RTreeObject for FragmentEntry {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

fn rect_to_aabb(r: Rect<f64>) -> AABB<[f64; 2]> {
    AABB::from_corners([r.min().x, r.min().y], [r.max().x, r.max().y])
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Partition a slice of (possibly overlapping) polygons into the maximal set of
/// non-overlapping atomic fragments, and return for each original polygon the
/// sorted list of fragment indices it covers.
///
/// This mirrors `geometry.fragmentize()` in Python:
///  1. Extract boundary ring(s) of every image polygon → GEOS linestrings.
///  2. `unary_union` the boundaries (noding), then `GEOSPolygonize`.
///  3. Build an R-tree on the ~20k fragments.
///  4. For each of the ~200 images: bbox query the R-tree to get candidates,
///     then do an exact `PreparedGeometry::covers` check per candidate.
///
/// Complexity:
///   * Build:  O(F log F)       — R-tree bulk-load on F fragments
///   * Query:  O(I × (log F + K)) — I images, K avg candidates per image
///
/// At 200 images × 20k fragments this is ~3-4 orders of magnitude fewer
/// predicate evaluations than the naïve O(I × F) nested loop.
pub fn fragmentize(
    images: &[Polygon<f64>],
) -> Result<(Vec<Polygon<f64>>, Vec<Vec<usize>>), geos::Error> {
    if images.is_empty() {
        return Ok((vec![], vec![]));
    }

    // ── 1. Convert images to GEOS; extract boundaries ─────────────────────
    let image_geoms: Vec<GeosGeometry> = images
        .iter()
        .map(GeosGeometry::try_from)
        .collect::<Result<_, _>>()?;

    let boundary_geoms: Vec<GeosGeometry> = image_geoms
        .iter()
        .map(|g| g.boundary())
        .collect::<Result<_, _>>()?;

    // ── 2. Node boundaries then polygonize ────────────────────────────────
    // unary_union merges/nodes all boundary linestrings so that intersection
    // points become explicit vertices — identical to Shapely's approach.
    let boundary_collection = GeosGeometry::create_geometry_collection(boundary_geoms)?;
    let noded = boundary_collection.unary_union()?;
    let poly_collection = GeosGeometry::polygonize(&[noded])?;

    let n_frags = poly_collection.get_num_geometries()?;
    if n_frags == 0 {
        return Ok((vec![], vec![vec![]; images.len()]));
    }

    // ── 3. Extract fragment polygons and their GEOS counterparts ──────────
    let mut fragment_geoms: Vec<GeosGeometry> = Vec::with_capacity(n_frags);
    let mut fragments: Vec<Polygon<f64>> = Vec::with_capacity(n_frags);

    for i in 0..n_frags {
        let geos_frag = poly_collection.get_geometry_n(i)?;
        let geo_geom = GeoGeometry::<f64>::try_from(geos_frag)?;
        let poly = match geo_geom {
            GeoGeometry::Polygon(p) => p,
            other => {
                return Err(geos::Error::GenericError(format!(
                    "polygonize returned non-polygon: {:?}",
                    std::mem::discriminant(&other)
                )))
            }
        };
        fragment_geoms.push(GeosGeometry::try_from(&poly)?);
        fragments.push(poly);
    }

    // ── 4. Build R-tree on fragments (bulk-load = O(F log F)) ─────────────
    let entries: Vec<FragmentEntry> = fragments
        .iter()
        .enumerate()
        .map(|(idx, poly)| FragmentEntry {
            idx,
            envelope: rect_to_aabb(poly.bounding_rect().expect("non-degenerate fragment")),
        })
        .collect();

    let rtree: RTree<FragmentEntry> = RTree::bulk_load(entries);

    // ── 5. Per-image: buffered PreparedGeometry + R-tree query ───────────
    // PreparedGeometry is built once per image and reused for all candidates.
    // Buffer by 1e-9 to handle boundary-coincident fragments (mirrors Python).
    let mut images_to_fragments: Vec<Vec<usize>> = vec![Vec::new(); images.len()];

    for (img_idx, (image, image_geom)) in images.iter().zip(image_geoms.iter()).enumerate() {
        // Expand image bbox by the same 1e-9 buffer so the R-tree query
        // returns any fragment that might be covered after buffering.
        let img_rect = image.bounding_rect().expect("non-degenerate image");
        let query_env = AABB::from_corners(
            [img_rect.min().x - 1e-9, img_rect.min().y - 1e-9],
            [img_rect.max().x + 1e-9, img_rect.max().y + 1e-9],
        );

        // R-tree narrows the candidate set: O(log F + K)
        let candidates: Vec<usize> = rtree
            .locate_in_envelope_intersecting(&query_env)
            .map(|e| e.idx)
            .collect();

        if candidates.is_empty() {
            continue;
        }

        // Build PreparedGeometry once for this image (amortised over candidates)
        let buffered = image_geom.buffer(1e-9, 3)?;
        let prepared = buffered.to_prepared_geom()?;

        for frag_idx in candidates {
            if prepared.covers(&fragment_geoms[frag_idx])? {
                images_to_fragments[img_idx].push(frag_idx);
            }
        }

        images_to_fragments[img_idx].sort_unstable();
    }

    Ok((fragments, images_to_fragments))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use geo::polygon;

    /// Two overlapping squares → 3 fragments (left-only, overlap, right-only).
    #[test]
    fn test_two_overlapping_squares() {
        let left = polygon![
            (x: 0.0, y: 0.0), (x: 2.0, y: 0.0),
            (x: 2.0, y: 2.0), (x: 0.0, y: 2.0),
            (x: 0.0, y: 0.0),
        ];
        let right = polygon![
            (x: 1.0, y: 0.0), (x: 3.0, y: 0.0),
            (x: 3.0, y: 2.0), (x: 1.0, y: 2.0),
            (x: 1.0, y: 0.0),
        ];

        let (fragments, img_to_frag) = fragmentize(&[left, right]).unwrap();

        assert_eq!(fragments.len(), 3, "expected 3 fragments");
        assert_eq!(img_to_frag[0].len(), 2, "left image covers 2 fragments");
        assert_eq!(img_to_frag[1].len(), 2, "right image covers 2 fragments");

        let shared: Vec<usize> = img_to_frag[0]
            .iter()
            .filter(|f| img_to_frag[1].contains(f))
            .cloned()
            .collect();
        assert_eq!(shared.len(), 1, "exactly one shared fragment");
    }

    /// Three non-overlapping squares → 3 fragments, one per image.
    #[test]
    fn test_non_overlapping() {
        let a = polygon![
            (x: 0.0, y: 0.0), (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0), (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ];
        let b = polygon![
            (x: 2.0, y: 0.0), (x: 3.0, y: 0.0),
            (x: 3.0, y: 1.0), (x: 2.0, y: 1.0),
            (x: 2.0, y: 0.0),
        ];
        let c = polygon![
            (x: 4.0, y: 0.0), (x: 5.0, y: 0.0),
            (x: 5.0, y: 1.0), (x: 4.0, y: 1.0),
            (x: 4.0, y: 0.0),
        ];

        let (fragments, img_to_frag) = fragmentize(&[a, b, c]).unwrap();

        assert_eq!(fragments.len(), 3);
        for frags in &img_to_frag {
            assert_eq!(frags.len(), 1);
        }
        assert_ne!(img_to_frag[0][0], img_to_frag[1][0]);
        assert_ne!(img_to_frag[1][0], img_to_frag[2][0]);
    }

    /// One image containing another entirely → 2 fragments.
    #[test]
    fn test_one_contained_in_other() {
        let outer = polygon![
            (x: 0.0, y: 0.0), (x: 4.0, y: 0.0),
            (x: 4.0, y: 4.0), (x: 0.0, y: 4.0),
            (x: 0.0, y: 0.0),
        ];
        let inner = polygon![
            (x: 1.0, y: 1.0), (x: 3.0, y: 1.0),
            (x: 3.0, y: 3.0), (x: 1.0, y: 3.0),
            (x: 1.0, y: 1.0),
        ];

        let (fragments, img_to_frag) = fragmentize(&[outer, inner]).unwrap();

        // Outer - inner ring + inner itself = 2 fragments
        assert_eq!(fragments.len(), 2, "expected 2 fragments: ring and inner");
        // outer covers both fragments, inner covers only the inner fragment
        assert_eq!(img_to_frag[0].len(), 2, "outer image covers both fragments");
        assert_eq!(img_to_frag[1].len(), 1, "inner image covers 1 fragment");
    }
}
