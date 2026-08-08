//! GEOS-backed `fragmentize` — optional reference/cross-check backend.
//!
//! Requires system libgeos (GEOS >= 3.14) and the `geos` feature. The default
//! [`crate::fragmentize`] is pure Rust and needs neither. Kept for parity
//! testing against the arrangement produced by `i_overlay`.
use geo::{BoundingRect, Geometry as GeoGeometry, InteriorPoint, Point, Polygon};
use geos::{Geom, Geometry as GeosGeometry};
use rayon::prelude::*;
use rstar::{AABB, RTree, RTreeObject};
use std::convert::TryFrom;

// ── Spatial index entry ───────────────────────────────────────────────────────

/// A fragment's interior representative point, indexed in an R-tree by fragment
/// index. Point-in-polygon (rather than polygon-covers) is exact here because a
/// fragment — being a cell of the arrangement of all image boundaries — never
/// straddles an image edge: it lies wholly inside or wholly outside each image,
/// so testing a single interior point decides coverage.
struct FragmentPoint {
    idx: usize,
    point: [f64; 2],
}

impl RTreeObject for FragmentPoint {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Partition a slice of (possibly overlapping) polygons into the maximal set of
/// non-overlapping atomic fragments, and return for each original polygon the
/// sorted list of fragment indices it covers.
///
/// This mirrors `geometry.fragmentize()` in Python:
///  1. Extract boundary ring(s) of every image polygon → GEOS linestrings.
///  2. `unary_union` the boundaries (noding), then `GEOSPolygonize`.
///  3. Take each fragment's interior representative point (`interior_point`)
///     and index those points in an R-tree.
///  4. For each image: build a `PreparedGeometry` once, query the R-tree with
///     the image bbox for candidate fragment points, and keep those the image
///     `contains` (a prepared point-in-polygon test).
///
/// Complexity:
///   * Build:  O(F log F)       — R-tree bulk-load on F fragment points
///   * Query:  O(I × (log F + K)) — I images, K avg candidates per image
///
/// Using a prepared point-in-polygon test instead of a buffered polygon-covers
/// check keeps each predicate evaluation cheap even in the dense-overlap regime
/// (biobjective pools: ~1000 images all spanning the AOI), where the R-tree
/// bbox prune degenerates and K approaches F.
pub fn fragmentize(
    images: &[Polygon<f64>],
) -> Result<(Vec<Polygon<f64>>, Vec<Vec<usize>>), geos::Error> {
    if images.is_empty() {
        return Ok((vec![], vec![]));
    }

    // Optional phase timing (set SIMS_FRAG_TIMING=1).
    let timing = std::env::var_os("SIMS_FRAG_TIMING").is_some();
    let mut t = std::time::Instant::now();
    let mut lap = |label: &str| {
        if timing {
            eprintln!("    [frag] {label}: {:.3} s", t.elapsed().as_secs_f64());
            t = std::time::Instant::now();
        }
    };

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
    lap("union+polygonize");

    // ── 3. Extract fragment polygons + interior representative points ─────
    // Convert the whole polygonized collection to geo in one shot (one FFI
    // walk instead of 5 FFI calls × F fragments), then compute each fragment's
    // interior representative point with geo's pure-Rust `interior_point` in
    // parallel. Any strictly-interior point decides coverage (step 5), so this
    // is equivalent to GEOS `point_on_surface` but needs no per-fragment FFI.
    let geo_collection = match GeoGeometry::<f64>::try_from(poly_collection)? {
        GeoGeometry::GeometryCollection(gc) => gc,
        other => {
            return Err(geos::Error::GenericError(format!(
                "polygonize returned non-collection: {:?}",
                std::mem::discriminant(&other)
            )));
        }
    };

    let (fragments, frag_points): (Vec<Polygon<f64>>, Vec<FragmentPoint>) = geo_collection
        .0
        .into_par_iter()
        .enumerate()
        .map(|(i, g)| {
            let poly = match g {
                GeoGeometry::Polygon(p) => p,
                other => {
                    return Err(geos::Error::GenericError(format!(
                        "polygonize returned non-polygon: {:?}",
                        std::mem::discriminant(&other)
                    )));
                }
            };
            let ip: Point<f64> = poly
                .interior_point()
                .expect("non-degenerate fragment has an interior point");
            Ok((
                poly,
                FragmentPoint {
                    idx: i,
                    point: [ip.x(), ip.y()],
                },
            ))
        })
        .collect::<Result<Vec<_>, geos::Error>>()?
        .into_iter()
        .unzip();

    lap("extract fragments");

    // ── 4. Build R-tree on fragment points (bulk-load = O(F log F)) ───────
    let rtree: RTree<FragmentPoint> = RTree::bulk_load(frag_points);
    lap("build rtree");

    // ── 5. Per-image: PreparedGeometry + point-in-polygon over candidates ─
    // Images are independent, so this is done in parallel. geos uses a
    // thread-local context, so each rayon worker builds its own GEOS objects;
    // the R-tree holds plain coordinate data and is shared read-only. The
    // pre-built `image_geoms` are not `Send`, so each task re-converts its
    // image polygon locally (cheap: one geometry per image).
    let images_to_fragments: Vec<Vec<usize>> = images
        .par_iter()
        .map(|image| -> Result<Vec<usize>, geos::Error> {
            let img_rect = image.bounding_rect().expect("non-degenerate image");
            let query_env = AABB::from_corners(
                [img_rect.min().x, img_rect.min().y],
                [img_rect.max().x, img_rect.max().y],
            );

            // Build PreparedGeometry once for this image (amortised over candidates).
            let geom = GeosGeometry::try_from(image)?;
            let prepared = geom.to_prepared_geom()?;

            // R-tree narrows the candidate set: O(log F + K). Iterate lazily to
            // avoid materialising the (potentially F-sized) candidate list.
            let mut assigned = Vec::new();
            for entry in rtree.locate_in_envelope_intersecting(&query_env) {
                if prepared.contains_xy(entry.point[0], entry.point[1])? {
                    assigned.push(entry.idx);
                }
            }
            assigned.sort_unstable();
            Ok(assigned)
        })
        .collect::<Result<_, _>>()?;
    lap("assign (parallel)");

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
