//! Coverage-aware instance-size sampler — Rust port of the Python
//! `generate_instances.py` that produced the publication image sets.
//!
//! Given a pool of candidate image footprints over an AOI and a target size `n`,
//! it selects `n` images that (a) fully cover the AOI and (b) spread coverage as
//! evenly as possible, with a small bias toward lower cloud. It is randomized
//! (seeded) so each `(instance, size)` is generated independently rather than by
//! nesting/downsampling.
//!
//! Pipeline (mirrors the reference implementation):
//! 1. **Grid** the AOI bounding box into cell centres, keep those inside the AOI.
//! 2. **Coverage sets**: for each image, the bitset of grid cells it covers.
//! 3. **Per-size subsample**: a seeded random subset (~`keep_frac` of the pool)
//!    that still covers every cell — makes different sizes non-nested.
//! 4. **Backbone**: randomized greedy set-cover to reach 100% coverage.
//! 5. **Even fill** to `n`: repeatedly add the image that most raises the
//!    least-covered cells (coverage-redundancy weight `1/(1+redun)²`), with a
//!    rank-weighted restricted candidate list (RCL) for randomization.
//!
//! Performance: coverage sets are `FixedBitSet`s built in parallel (rayon) with
//! an R-tree bounding-box prefilter over the grid; the greedy loops are
//! word-level bitset ops.

use fixedbitset::FixedBitSet;
use geo::{BoundingRect, Contains, Coord, MultiPolygon, Point, Polygon};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rstar::{AABB, RTree, RTreeObject};

type P2 = [f64; 2];

/// Sampler parameters. `Default` reproduces the reference values.
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    /// Number of grid cells along the longer AOI axis.
    pub grid_long: usize,
    /// Restricted-candidate-list size for the randomized greedy pick.
    pub rcl_k: usize,
    /// Per-size random subsample fraction of the pool (independence across sizes).
    pub keep_frac: f64,
    /// Small pull toward lower cloud coverage in the selection score.
    pub cloud_tiebreak: f64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            grid_long: 55,
            rcl_k: 8,
            keep_frac: 0.6,
            cloud_tiebreak: 1e-3,
        }
    }
}

/// Result of a selection.
#[derive(Clone, Debug)]
pub struct Selection {
    /// Selected image indices, in the order they were picked.
    pub order: Vec<usize>,
    /// Number of images in the coverage backbone (prefix of `order`).
    pub backbone_size: usize,
    /// Total number of AOI grid cells.
    pub num_cells: usize,
    /// Cells covered by the selection (== `num_cells` on success).
    pub covered_cells: usize,
    /// Per-cell coverage count under the final selection (evenness diagnostics).
    pub redundancy: Vec<u32>,
}

impl Selection {
    /// The size-`n` instance: the first `n` picks. When the coverage backbone
    /// alone already exceeds `n`, `order` is longer than `n` and this truncates
    /// it (mirroring the reference `order[:n]`).
    #[must_use]
    pub fn image_indices(&self, n: usize) -> &[usize] {
        &self.order[..n.min(self.order.len())]
    }
}

/// A grid cell centre carrying its index, for the R-tree prefilter.
struct CellPoint {
    point: P2,
    idx: usize,
}
impl RTreeObject for CellPoint {
    type Envelope = AABB<P2>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point)
    }
}

/// AOI grid-cell centres that fall inside the AOI polygon.
#[must_use]
pub fn grid_cells(aoi: &Polygon<f64>, grid_long: usize) -> Vec<Coord<f64>> {
    let Some(rect) = aoi.bounding_rect() else {
        return Vec::new();
    };
    let (minx, miny) = (rect.min().x, rect.min().y);
    let (w, h) = (rect.max().x - minx, rect.max().y - miny);
    if w <= 0.0 || h <= 0.0 || grid_long == 0 {
        return Vec::new();
    }
    let gl = grid_long as f64;
    let (nx, ny) = if w >= h {
        (grid_long, ((gl * h / w).round() as usize).max(1))
    } else {
        (((gl * w / h).round() as usize).max(1), grid_long)
    };
    let mut cells = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y = miny + (iy as f64 + 0.5) * h / ny as f64;
        for ix in 0..nx {
            let x = minx + (ix as f64 + 0.5) * w / nx as f64;
            let c = Coord { x, y };
            if aoi.contains(&Point::from(c)) {
                cells.push(c);
            }
        }
    }
    cells
}

/// For each image, the bitset of grid cells whose centre it covers.
///
/// Built in parallel; an R-tree over the cells restricts the point-in-polygon
/// tests to each image's bounding box.
#[must_use]
pub fn coverage_bitsets(images: &[MultiPolygon<f64>], cells: &[Coord<f64>]) -> Vec<FixedBitSet> {
    let n_cells = cells.len();
    let tree = RTree::bulk_load(
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| CellPoint {
                point: [c.x, c.y],
                idx: i,
            })
            .collect(),
    );
    images
        .par_iter()
        .map(|mp| {
            let mut bs = FixedBitSet::with_capacity(n_cells);
            if let Some(rect) = mp.bounding_rect() {
                let env =
                    AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);
                for cp in tree.locate_in_envelope_intersecting(&env) {
                    let c = cells[cp.idx];
                    if mp.contains(&Point::new(c.x, c.y)) {
                        bs.insert(cp.idx);
                    }
                }
            }
            bs
        })
        .collect()
}

/// Number of set bits `a` has that are **not** set in `covered`.
#[inline]
fn new_cover_count(a: &FixedBitSet, covered: &FixedBitSet) -> usize {
    a.ones().filter(|&c| !covered.contains(c)).count()
}

/// Randomized greedy pick: choose among the top-`k` highest-scoring non-blocked
/// candidates with rank weights `k, k-1, …, 1`. Returns `None` if none remain.
fn pick_rcl(scores: &[f64], blocked: &[bool], k_max: usize, rng: &mut impl Rng) -> Option<usize> {
    let mut cand: Vec<(f64, usize)> = (0..scores.len())
        .filter(|&i| !blocked[i] && scores[i].is_finite())
        .map(|i| (scores[i], i))
        .collect();
    if cand.is_empty() {
        return None;
    }
    // Descending by score; ties broken by lower index (deterministic given RNG).
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let k = k_max.min(cand.len());
    let total = (k * (k + 1) / 2) as f64; // Σ 1..=k
    let r = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (j, &(_, idx)) in cand.iter().take(k).enumerate() {
        cum += (k - j) as f64; // weight of rank j is (k-j)
        if r < cum {
            return Some(idx);
        }
    }
    Some(cand[k - 1].1)
}

/// A seeded random subset (~`keep_frac` of the pool) that still covers every
/// cell and has at least `n` images. Grows only if coverage/size demands it.
fn subsample(
    coverage: &[FixedBitSet],
    num_cells: usize,
    n: usize,
    keep_frac: f64,
    rng: &mut impl Rng,
) -> Vec<bool> {
    let n_img = coverage.len();
    let mut perm: Vec<usize> = (0..n_img).collect();
    perm.shuffle(rng);
    let mut size = ((n_img as f64 * keep_frac).round() as usize)
        .max(n)
        .min(n_img);
    let step = (n_img / 50).max(1);
    loop {
        let mut cov = FixedBitSet::with_capacity(num_cells);
        for &i in &perm[..size] {
            cov.union_with(&coverage[i]);
        }
        if cov.count_ones(..) == num_cells || size == n_img {
            let mut avail = vec![false; n_img];
            for &i in &perm[..size] {
                avail[i] = true;
            }
            return avail;
        }
        size = (size + step).min(n_img);
    }
}

/// Select `n` images from `coverage` (over `num_cells` grid cells) with the
/// backbone + even-fill strategy. `cloud[i]` is image `i`'s cloud coverage %.
///
/// # Panics
/// Panics if `coverage.len() != cloud.len()`.
#[must_use]
pub fn select(
    coverage: &[FixedBitSet],
    num_cells: usize,
    cloud: &[f64],
    n: usize,
    params: SampleParams,
    rng: &mut impl Rng,
) -> Selection {
    assert_eq!(
        coverage.len(),
        cloud.len(),
        "coverage and cloud length mismatch"
    );
    let n_img = coverage.len();

    let avail = subsample(coverage, num_cells, n, params.keep_frac, rng);
    let mut blocked: Vec<bool> = avail.iter().map(|&a| !a).collect();
    let mut covered = FixedBitSet::with_capacity(num_cells);
    let mut redun = vec![0u32; num_cells];
    let mut order = Vec::with_capacity(n);

    // 1. Coverage backbone: randomized greedy set-cover within the subsample.
    while covered.count_ones(..) < num_cells {
        let mut scores = vec![f64::NEG_INFINITY; n_img];
        let mut any_gain = false;
        for i in 0..n_img {
            if blocked[i] {
                continue;
            }
            let g =
                new_cover_count(&coverage[i], &covered) as f64 - params.cloud_tiebreak * cloud[i];
            scores[i] = g;
            if g > 0.0 {
                any_gain = true;
            }
        }
        if !any_gain {
            break;
        }
        let Some(i) = pick_rcl(&scores, &blocked, params.rcl_k, rng) else {
            break;
        };
        blocked[i] = true;
        order.push(i);
        for c in coverage[i].ones() {
            covered.insert(c);
            redun[c] += 1;
        }
    }
    let backbone_size = order.len();

    // 2. Even fill to `n`: raise the least-covered cells the most.
    while order.len() < n {
        let weight: Vec<f64> = redun
            .iter()
            .map(|&r| {
                let d = 1.0 + f64::from(r);
                1.0 / (d * d)
            })
            .collect();
        let mut scores = vec![f64::NEG_INFINITY; n_img];
        for i in 0..n_img {
            if blocked[i] {
                continue;
            }
            let s: f64 = coverage[i].ones().map(|c| weight[c]).sum::<f64>()
                - params.cloud_tiebreak * cloud[i];
            scores[i] = s;
        }
        let Some(i) = pick_rcl(&scores, &blocked, params.rcl_k, rng) else {
            break;
        };
        blocked[i] = true;
        order.push(i);
        for c in coverage[i].ones() {
            redun[c] += 1;
        }
    }

    Selection {
        order,
        backbone_size,
        num_cells,
        covered_cells: covered.count_ones(..),
        redundancy: redun,
    }
}

/// End-to-end: grid the AOI, build coverage, and select `n` images. `seed` makes
/// the (deterministic) selection reproducible.
#[must_use]
pub fn sample_instance(
    aoi: &Polygon<f64>,
    images: &[MultiPolygon<f64>],
    cloud: &[f64],
    n: usize,
    params: SampleParams,
    seed: u64,
) -> Selection {
    let cells = grid_cells(aoi, params.grid_long);
    let coverage = coverage_bitsets(images, &cells);
    let mut rng = StdRng::seed_from_u64(seed);
    select(&coverage, cells.len(), cloud, n, params, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{LineString, Polygon};

    fn unit_square() -> Polygon<f64> {
        Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![],
        )
    }

    /// Axis-aligned rectangle as a MultiPolygon.
    fn rect_mp(x0: f64, y0: f64, x1: f64, y1: f64) -> MultiPolygon<f64> {
        MultiPolygon::new(vec![Polygon::new(
            LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
            vec![],
        )])
    }

    #[test]
    fn grid_cells_inside_aoi() {
        let cells = grid_cells(&unit_square(), 10);
        assert!(!cells.is_empty());
        for c in &cells {
            assert!(c.x > 0.0 && c.x < 10.0 && c.y > 0.0 && c.y < 10.0);
        }
    }

    #[test]
    fn coverage_split_halves() {
        let cells = grid_cells(&unit_square(), 10);
        let left = rect_mp(0.0, 0.0, 5.0, 10.0);
        let right = rect_mp(5.0, 0.0, 10.0, 10.0);
        let cov = coverage_bitsets(&[left, right], &cells);
        // Every cell is in exactly one half; together they cover all cells.
        let mut all = FixedBitSet::with_capacity(cells.len());
        all.union_with(&cov[0]);
        all.union_with(&cov[1]);
        assert_eq!(all.count_ones(..), cells.len());
        assert_eq!(cov[0].intersection(&cov[1]).count(), 0);
    }

    #[test]
    fn selects_full_cover_and_is_deterministic() {
        let aoi = unit_square();
        // Large, heavily overlapping tiles so the backbone is small (< n) and the
        // even-fill reaches n.
        let imgs: Vec<MultiPolygon<f64>> = (0..12)
            .map(|k| {
                let x = (k % 3) as f64 * 2.0;
                let y = (k / 3) as f64 * 2.0;
                rect_mp(x - 2.0, y - 2.0, x + 8.0, y + 8.0)
            })
            .collect();
        let cloud: Vec<f64> = (0..imgs.len()).map(|k| (k * 7 % 100) as f64).collect();

        let a = sample_instance(&aoi, &imgs, &cloud, 6, SampleParams::default(), 42);
        let b = sample_instance(&aoi, &imgs, &cloud, 6, SampleParams::default(), 42);
        assert_eq!(a.order, b.order, "same seed must give same selection");
        assert_eq!(a.covered_cells, a.num_cells, "selection must cover the AOI");
        assert_eq!(a.image_indices(6).len(), 6);
        assert!(a.backbone_size >= 1 && a.backbone_size <= a.order.len());
        // distinct picks
        let mut s = a.order.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), a.order.len(), "no image picked twice");
    }

    #[test]
    fn lower_cloud_preferred_on_ties() {
        // Two identical full-cover images; the lower-cloud one should be picked.
        let aoi = unit_square();
        let full_a = rect_mp(-1.0, -1.0, 11.0, 11.0);
        let full_b = rect_mp(-1.0, -1.0, 11.0, 11.0);
        let cov = coverage_bitsets(&[full_a, full_b], &grid_cells(&aoi, 10));
        let cells = grid_cells(&aoi, 10).len();
        let mut rng = StdRng::seed_from_u64(0);
        // image 1 has far lower cloud; backbone should take it first.
        let sel = select(
            &cov,
            cells,
            &[90.0, 1.0],
            1,
            SampleParams {
                rcl_k: 1,
                ..Default::default()
            },
            &mut rng,
        );
        assert_eq!(sel.order.first(), Some(&1));
    }
}
