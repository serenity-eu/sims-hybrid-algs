//! Pure-Rust `fragmentize` — no system libgeos (default backend).
//!
//! Same contract as the optional [`crate::fragmentize_geos`] backend, but the
//! planar arrangement (GEOS `unary_union` + `polygonize`) is replaced by
//! `i_overlay`'s slice operation, and all predicates use `geo`.
//!
//! Two entry points:
//!  * [`fragmentize`] — arrange a set of already-clipped images.
//!  * [`fragmentize_within_aoi`] — clip to an AOI and arrange in one pass.
//!
//! Approach:
//!  1. Slice the images (as the `NonZero`-filled subject) by all their own
//!     boundaries; each image edge becomes a cut, yielding the atomic fragments.
//!     `fragmentize_within_aoi` additionally throws the AOI boundary into the
//!     same slice and drops fragments outside the AOI, so the clip shares the
//!     arrangement's single integer grid (no double-snap).
//!  2. Compute each fragment's interior representative point (`geo`).
//!  3. Index the points in an R-tree and, per image, keep the fragments whose
//!     representative point falls inside the image (inline ray-cast test).

use geo::{BoundingRect, Contains, InteriorPoint, LineString, MultiPolygon, Point, Polygon};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::float::slice::FloatSlice;
use rayon::prelude::*;
use rstar::{AABB, RTree, RTreeObject};

type P2 = [f64; 2];

/// The atomic fragment polygons and, for each image, the sorted indices of the
/// fragments it covers.
#[derive(Debug, Clone, Default)]
pub struct Fragmentation {
    /// The atomic, non-overlapping fragment polygons.
    pub fragments: Vec<Polygon<f64>>,
    /// Per image, the sorted indices (into `fragments`) of the fragments it covers.
    pub images_to_fragments: Vec<Vec<usize>>,
}

/// [`Fragmentation`] plus, for each image, the fragments that are cloudy in it.
#[derive(Debug, Clone, Default)]
pub struct CloudFragmentation {
    /// The atomic, non-overlapping fragment polygons.
    pub fragments: Vec<Polygon<f64>>,
    /// Per image, the sorted indices (into `fragments`) of the fragments it covers.
    pub images_to_fragments: Vec<Vec<usize>>,
    /// Per image, the sorted indices (into `fragments`) of the fragments that are
    /// cloudy in it — the SIMS `clouds[image]` set (fragment ⊆ image ∩ its cloud).
    pub images_to_cloudy_fragments: Vec<Vec<usize>>,
}

struct FragmentPoint {
    idx: usize,
    point: P2,
}

impl RTreeObject for FragmentPoint {
    type Envelope = AABB<P2>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point)
    }
}

/// Even-odd ray-cast point-in-ring test. Much cheaper per call than routing
/// through `geo::Contains` (which goes via the general `Relate` machinery) —
/// and the assignment step runs this hundreds of millions of times on the
/// dense pools.
#[inline]
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
            let x_cross = xi + (y - yi) / (yj - yi) * (xj - xi);
            if x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Point-in-polygon: inside the exterior and outside every hole.
#[inline]
fn point_in_polygon(poly: &Polygon<f64>, x: f64, y: f64) -> bool {
    point_in_ring(poly.exterior(), x, y) && !poly.interiors().iter().any(|h| point_in_ring(h, x, y))
}

/// Convert a geo ring to an i_overlay path (drop the closing duplicate vertex,
/// which i_overlay represents implicitly).
fn ring_to_path(ring: &LineString<f64>) -> Vec<P2> {
    let mut pts: Vec<P2> = ring.coords().map(|c| [c.x, c.y]).collect();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    pts
}

/// Convert one i_overlay shape (outer contour + holes) to a geo Polygon.
fn shape_to_polygon(shape: &[Vec<P2>]) -> Option<Polygon<f64>> {
    let mut rings = shape
        .iter()
        .map(|c| LineString::from(c.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>()));
    let exterior = rings.next()?;
    let interiors: Vec<LineString<f64>> = rings.collect();
    Some(Polygon::new(exterior, interiors))
}

/// Collect all boundary rings (exterior + holes) of `polys` as i_overlay paths.
fn rings_of(polys: &[Polygon<f64>]) -> Vec<Vec<P2>> {
    let mut paths = Vec::new();
    for p in polys {
        paths.push(ring_to_path(p.exterior()));
        for hole in p.interiors() {
            paths.push(ring_to_path(hole));
        }
    }
    paths
}

/// Turn a set of atomic fragment `shapes` (from an i_overlay slice) into geo
/// polygons plus an R-tree of their interior representative points. If `aoi` is
/// `Some`, fragments whose representative point lies outside it are dropped.
fn build_fragment_index(
    shapes: &[Vec<Vec<P2>>],
    aoi: Option<&Polygon<f64>>,
) -> (Vec<Polygon<f64>>, RTree<FragmentPoint>) {
    let (fragments, frag_points): (Vec<Polygon<f64>>, Vec<FragmentPoint>) = shapes
        .par_iter()
        .filter_map(|shape| {
            let poly = shape_to_polygon(shape)?;
            let ip: Point<f64> = poly.interior_point()?;
            if let Some(aoi) = aoi
                && !point_in_polygon(aoi, ip.x(), ip.y())
            {
                return None; // fragment outside the AOI (part of an image beyond it)
            }
            Some((
                poly,
                FragmentPoint {
                    idx: 0,
                    point: [ip.x(), ip.y()],
                },
            )) // idx fixed up below
        })
        .unzip();
    // Reindex so each point's `idx` matches its fragment's position.
    let frag_points: Vec<FragmentPoint> = frag_points
        .into_iter()
        .enumerate()
        .map(|(idx, fp)| FragmentPoint {
            idx,
            point: fp.point,
        })
        .collect();
    (fragments, RTree::bulk_load(frag_points))
}

/// Fragments of `rtree` whose representative point lies inside `image`.
fn assign_fragments(rtree: &RTree<FragmentPoint>, image: &Polygon<f64>) -> Vec<usize> {
    let Some(rect) = image.bounding_rect() else {
        return Vec::new();
    };
    let env = AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);
    let mut assigned: Vec<usize> = rtree
        .locate_in_envelope_intersecting(&env)
        .filter(|e| point_in_polygon(image, e.point[0], e.point[1]))
        .map(|e| e.idx)
        .collect();
    assigned.sort_unstable();
    assigned
}

/// Turn fragment `shapes` into geo polygons and, for each image, the sorted
/// indices of the fragments it covers. Shared by [`fragmentize`] and
/// [`fragmentize_within_aoi`].
fn assemble(
    shapes: &[Vec<Vec<P2>>],
    images: &[Polygon<f64>],
    aoi: Option<&Polygon<f64>>,
) -> Fragmentation {
    let (fragments, rtree) = build_fragment_index(shapes, aoi);
    let images_to_fragments = images
        .par_iter()
        .map(|im| assign_fragments(&rtree, im))
        .collect();
    Fragmentation {
        fragments,
        images_to_fragments,
    }
}

/// Fragmentize a set of (already-clipped) image polygons into atomic
/// non-overlapping fragments, returning for each image the fragments it covers.
///
/// The fragments partition the *union* of the images.
pub fn fragmentize(images: &[Polygon<f64>]) -> Fragmentation {
    if images.is_empty() {
        return Fragmentation::default();
    }
    // Slice the union (NonZero fill over all rings) by all rings → fragments.
    let paths = rings_of(images);
    let shapes = paths.slice_by(&paths, FillRule::NonZero);
    assemble(&shapes, images, None)
}

/// Single-pass clip + fragmentize: partition the AOI into atomic fragments cut
/// by every (unclipped) image boundary, returning for each image the fragments
/// it covers.
///
/// Unlike clipping the images first and then calling [`fragmentize`], the clip
/// and the arrangement happen in **one** i_overlay pass on a **single** integer
/// grid. This avoids the double-snap that a separate i_overlay clip introduces
/// (degenerate zero-area slivers and small boundary misalignments), so a
/// fully pure-Rust pipeline stays sound without an exact (GEOS) pre-clip.
///
/// The AOI boundary is added to the arrangement so it cuts fragments too; parts
/// of images lying outside the AOI are then dropped. Fragments of the AOI
/// covered by no image survive with an empty entry in every image's list — i.e.
/// coverage holes surface rather than being silently dropped.
pub fn fragmentize_within_aoi(images: &[Polygon<f64>], aoi: &Polygon<f64>) -> Fragmentation {
    if images.is_empty() {
        return Fragmentation::default();
    }
    // Arrange all image boundaries *and* the AOI boundary together (images as
    // the filled subject so overlaps subdivide correctly), then keep only the
    // fragments inside the AOI. One pass, one grid — no double-snap.
    let mut paths = rings_of(images);
    paths.extend(rings_of(std::slice::from_ref(aoi)));
    let shapes = paths.slice_by(&paths, FillRule::NonZero);
    assemble(&shapes, images, Some(aoi))
}

/// Like [`fragmentize_within_aoi`], but each image's **cloud regions** are folded
/// into the same arrangement as an extra cutting layer, so every fragment is
/// atomic w.r.t. clouds too — a fragment lies entirely inside or entirely
/// outside each image's cloud. Returns `(fragments, images_to_fragments,
/// images_to_cloudy_fragments)`, where the third is, per image, the sorted
/// fragment indices that are cloudy in that image (fragment ⊆ image ∩ its cloud
/// region) — i.e. the SIMS `clouds[image]` set, computed *exactly* (no
/// fractional threshold).
///
/// `clouds[i]` is the cloud MultiPolygon for `images[i]` in lon/lat (e.g. from
/// [`crate::cloud::cloud_multipolygon`]); pass an empty MultiPolygon for a
/// cloud-free image. `clouds.len()` must equal `images.len()`.
///
/// Folding every image's cloud boundary into one arrangement can create many
/// more fragments than the cloud-free version; use it when the exact per-image
/// cloud/clear partition is needed.
pub fn fragmentize_within_aoi_with_clouds(
    images: &[Polygon<f64>],
    aoi: &Polygon<f64>,
    clouds: &[MultiPolygon<f64>],
) -> CloudFragmentation {
    assert_eq!(
        images.len(),
        clouds.len(),
        "one cloud MultiPolygon per image"
    );
    if images.is_empty() {
        return CloudFragmentation::default();
    }
    // Arrangement layers: image boundaries + AOI boundary + every cloud boundary.
    let mut paths = rings_of(images);
    paths.extend(rings_of(std::slice::from_ref(aoi)));
    for mp in clouds {
        paths.extend(rings_of(&mp.0));
    }
    let shapes = paths.slice_by(&paths, FillRule::NonZero);
    let (fragments, rtree) = build_fragment_index(&shapes, Some(aoi));

    // Per image: fragments inside it, and (subset) those also inside its cloud.
    let (images_to_fragments, images_to_cloudy_fragments): (Vec<_>, Vec<_>) = images
        .par_iter()
        .zip(clouds.par_iter())
        .map(|(image, cloud)| {
            let Some(rect) = image.bounding_rect() else {
                return (Vec::new(), Vec::new());
            };
            let env =
                AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);
            let (mut assigned, mut cloudy): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
            for e in rtree.locate_in_envelope_intersecting(&env) {
                let (x, y) = (e.point[0], e.point[1]);
                if point_in_polygon(image, x, y) {
                    assigned.push(e.idx);
                    if cloud.contains(&Point::new(x, y)) {
                        cloudy.push(e.idx);
                    }
                }
            }
            assigned.sort_unstable();
            cloudy.sort_unstable();
            (assigned, cloudy)
        })
        .unzip();
    CloudFragmentation {
        fragments,
        images_to_fragments,
        images_to_cloudy_fragments,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Area, polygon};

    /// An image covering the whole AOI with a central cloud: folding the cloud
    /// into the arrangement splits the AOI into a cloud fragment and a clear
    /// ring, and only the cloud fragment is reported cloudy for that image.
    #[test]
    fn test_fragmentize_with_clouds() {
        let aoi = polygon![(x:0.0,y:0.0),(x:4.0,y:0.0),(x:4.0,y:4.0),(x:0.0,y:4.0)];
        let image = polygon![(x:-1.0,y:-1.0),(x:5.0,y:-1.0),(x:5.0,y:5.0),(x:-1.0,y:5.0)];
        let cloud = MultiPolygon::new(vec![
            polygon![(x:1.0,y:1.0),(x:3.0,y:1.0),(x:3.0,y:3.0),(x:1.0,y:3.0)],
        ]);

        let CloudFragmentation {
            fragments: frags,
            images_to_fragments: i2f,
            images_to_cloudy_fragments: i2c,
        } = fragmentize_within_aoi_with_clouds(&[image], &aoi, &[cloud]);
        assert_eq!(frags.len(), 2, "AOI split into cloud + clear ring");
        assert_eq!(i2f[0].len(), 2, "image covers both fragments");
        assert_eq!(i2c[0].len(), 1, "exactly one cloudy fragment");
        let cloudy_area = frags[i2c[0][0]].unsigned_area();
        assert!(
            (3.5..=4.5).contains(&cloudy_area),
            "cloudy fragment ~4, got {cloudy_area}"
        );
    }

    /// A cloud-free image (empty MultiPolygon) yields no cloudy fragments.
    #[test]
    fn test_fragmentize_clouds_empty_is_clear() {
        let aoi = polygon![(x:0.0,y:0.0),(x:2.0,y:0.0),(x:2.0,y:2.0),(x:0.0,y:2.0)];
        let image = polygon![(x:-1.0,y:-1.0),(x:3.0,y:-1.0),(x:3.0,y:3.0),(x:-1.0,y:3.0)];
        let CloudFragmentation {
            images_to_fragments: i2f,
            images_to_cloudy_fragments: i2c,
            ..
        } = fragmentize_within_aoi_with_clouds(&[image], &aoi, &[MultiPolygon::new(vec![])]);
        assert!(!i2f[0].is_empty());
        assert!(i2c[0].is_empty(), "no cloud -> no cloudy fragments");
    }

    /// Two overlapping squares → 3 fragments (left-only, overlap, right-only),
    /// each image covering 2 of them and sharing exactly 1.
    #[test]
    fn test_two_overlapping_squares() {
        let left = polygon![
            (x: 0.0, y: 0.0), (x: 2.0, y: 0.0),
            (x: 2.0, y: 2.0), (x: 0.0, y: 2.0),
        ];
        let right = polygon![
            (x: 1.0, y: 0.0), (x: 3.0, y: 0.0),
            (x: 3.0, y: 2.0), (x: 1.0, y: 2.0),
        ];

        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize(&[left, right]);

        assert_eq!(fragments.len(), 3, "expected 3 fragments");
        assert_eq!(img_to_frag[0].len(), 2, "left covers 2 fragments");
        assert_eq!(img_to_frag[1].len(), 2, "right covers 2 fragments");
        let shared = img_to_frag[0]
            .iter()
            .filter(|f| img_to_frag[1].contains(f))
            .count();
        assert_eq!(shared, 1, "exactly one shared fragment");
    }

    /// Folded single-pass over the two overlapping squares inside an enclosing
    /// AOI: the AOI is subdivided into the 3 image cells plus the surrounding
    /// background cell. Each image still covers 2 cells sharing exactly 1, and
    /// the background cell is covered by no image.
    #[test]
    fn test_within_aoi_two_overlapping_squares() {
        let left = polygon![
            (x: 0.0, y: 0.0), (x: 2.0, y: 0.0),
            (x: 2.0, y: 2.0), (x: 0.0, y: 2.0),
        ];
        let right = polygon![
            (x: 1.0, y: 0.0), (x: 3.0, y: 0.0),
            (x: 3.0, y: 2.0), (x: 1.0, y: 2.0),
        ];
        let aoi = polygon![
            (x: -1.0, y: -1.0), (x: 4.0, y: -1.0),
            (x: 4.0, y: 3.0), (x: -1.0, y: 3.0),
        ];

        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize_within_aoi(&[left, right], &aoi);

        // 3 image cells + 1 background cell.
        assert_eq!(fragments.len(), 4, "expected 3 image cells + background");
        assert_eq!(img_to_frag[0].len(), 2, "left covers 2 cells");
        assert_eq!(img_to_frag[1].len(), 2, "right covers 2 cells");
        let shared = img_to_frag[0]
            .iter()
            .filter(|f| img_to_frag[1].contains(f))
            .count();
        assert_eq!(shared, 1, "exactly one shared cell");

        // Exactly one fragment (the background) is covered by no image.
        let mut covered = vec![false; fragments.len()];
        for frags in &img_to_frag {
            for &f in frags {
                covered[f] = true;
            }
        }
        assert_eq!(
            covered.iter().filter(|c| !**c).count(),
            1,
            "one background cell"
        );
    }

    /// A fully-covering image over the AOI yields a single fragment = the AOI.
    #[test]
    fn test_within_aoi_full_cover() {
        let aoi = polygon![(x: 0.0, y: 0.0), (x: 2.0, y: 0.0), (x: 2.0, y: 2.0), (x: 0.0, y: 2.0)];
        let img =
            polygon![(x: -1.0, y: -1.0), (x: 3.0, y: -1.0), (x: 3.0, y: 3.0), (x: -1.0, y: 3.0)];

        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize_within_aoi(&[img], &aoi);
        assert_eq!(fragments.len(), 1, "AOI is one cell");
        assert_eq!(img_to_frag[0].len(), 1, "image covers the whole AOI cell");
    }

    /// Three disjoint squares → 3 fragments, one per image.
    #[test]
    fn test_non_overlapping() {
        let a = polygon![(x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 1.0, y: 1.0), (x: 0.0, y: 1.0)];
        let b = polygon![(x: 2.0, y: 0.0), (x: 3.0, y: 0.0), (x: 3.0, y: 1.0), (x: 2.0, y: 1.0)];
        let c = polygon![(x: 4.0, y: 0.0), (x: 5.0, y: 0.0), (x: 5.0, y: 1.0), (x: 4.0, y: 1.0)];

        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize(&[a, b, c]);

        assert_eq!(fragments.len(), 3);
        for frags in &img_to_frag {
            assert_eq!(frags.len(), 1);
        }
    }

    /// One square fully containing another → 2 fragments (ring + inner); the
    /// outer image covers both, the inner covers only the inner fragment.
    #[test]
    fn test_one_contained_in_other() {
        let outer =
            polygon![(x: 0.0, y: 0.0), (x: 4.0, y: 0.0), (x: 4.0, y: 4.0), (x: 0.0, y: 4.0)];
        let inner =
            polygon![(x: 1.0, y: 1.0), (x: 3.0, y: 1.0), (x: 3.0, y: 3.0), (x: 1.0, y: 3.0)];

        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize(&[outer, inner]);

        assert_eq!(fragments.len(), 2, "expected ring + inner");
        assert_eq!(img_to_frag[0].len(), 2, "outer covers both");
        assert_eq!(img_to_frag[1].len(), 1, "inner covers one");
    }

    /// Empty input yields empty output.
    #[test]
    fn test_empty() {
        let Fragmentation {
            fragments,
            images_to_fragments: img_to_frag,
        } = fragmentize(&[]);
        assert!(fragments.is_empty());
        assert!(img_to_frag.is_empty());
    }
}
