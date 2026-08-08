//! Integration tests against real satellite data from publication-data/.
//!
//! These validate the default (pure-Rust) [`fragmentize`] backend against a
//! GEOS **oracle**: clipping and the invariant checks use exact double-precision
//! GEOS operations as ground truth. They therefore require the `geos` feature
//! (and system libgeos); the geos-free backend logic is covered by unit tests in
//! `src/fragmentize.rs`. Invariants checked per (AOI, image-set) pair:
//!
//! 1. **Coverage**: the union of all fragments covers the entire AOI.
//! 2. **No zero-cover fragment**: every fragment is covered by ≥1 image.
//! 3. **Image-fragment consistency**: for every image, the union of its
//!    assigned fragments matches the original (clipped) image polygon.
#![cfg(feature = "geos")]

use geo::{Area, Geometry as GeoGeometry, LineString, Polygon};
use geos::{Geom, Geometry as GeosGeometry};
use sims_preprocessing::fragmentize::{Fragmentation, fragmentize};
use std::convert::TryFrom;

// ── Paths ─────────────────────────────────────────────────────────────────────

const DATA_ROOT: &str = "../publication-data/satellite-data";

fn aoi_path(city: &str) -> String {
    format!("{DATA_ROOT}/aois/{city}.geojson")
}
fn images_path(city: &str, n: usize) -> String {
    format!("{DATA_ROOT}/images/{city}_{n}_images.geojson")
}

// ── GeoJSON helpers ───────────────────────────────────────────────────────────

/// Build a ring (LineString) from a GeoJSON coordinate ring `[[x,y],...]`.
fn ring_from_coords(coords: &serde_json::Value) -> Option<LineString<f64>> {
    let pts: Vec<(f64, f64)> = coords
        .as_array()?
        .iter()
        .filter_map(|c| {
            let a = c.as_array()?;
            Some((a[0].as_f64()?, a[1].as_f64()?))
        })
        .collect();
    Some(LineString::from(pts))
}

/// Build a geo::Polygon from a GeoJSON Polygon `coordinates` array.
fn polygon_from_coords(coords: &serde_json::Value) -> Option<Polygon<f64>> {
    let rings = coords.as_array()?;
    let exterior = ring_from_coords(rings.first()?)?;
    let interiors: Vec<LineString<f64>> = rings[1..].iter().filter_map(ring_from_coords).collect();
    Some(Polygon::new(exterior, interiors))
}

/// Load every Polygon feature from a GeoJSON FeatureCollection. MultiPolygons
/// contribute each of their component polygons (mirrors treating each part as a
/// footprint). Parses coordinates directly to avoid depending on the GEOS
/// GeoJSON reader (feature-gated behind v3_10_0).
fn load_polygons(path: &str) -> Vec<Polygon<f64>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let fc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut out = Vec::new();
    for feat in fc["features"].as_array().unwrap() {
        let geom = &feat["geometry"];
        match geom["type"].as_str() {
            Some("Polygon") => {
                if let Some(p) = polygon_from_coords(&geom["coordinates"]) {
                    out.push(p);
                }
            }
            Some("MultiPolygon") => {
                if let Some(parts) = geom["coordinates"].as_array() {
                    for part in parts {
                        if let Some(p) = polygon_from_coords(part) {
                            out.push(p);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Load the first (and typically only) polygon from a GeoJSON file.
fn load_aoi(path: &str) -> Polygon<f64> {
    let polys = load_polygons(path);
    assert!(!polys.is_empty(), "no polygon found in {path}");
    polys.into_iter().next().unwrap()
}

// ── Clipping ──────────────────────────────────────────────────────────────────

/// Clip each image to the AOI (mirrors Python's `clip(..., keep_geom_type=True)`).
/// Images that don't intersect the AOI or collapse to non-polygons are dropped.
fn clip_to_aoi(images: &[Polygon<f64>], aoi: &Polygon<f64>) -> Vec<Polygon<f64>> {
    let aoi_geom = GeosGeometry::try_from(aoi).expect("aoi to geos");

    images
        .iter()
        .filter_map(|img| {
            let img_geom = GeosGeometry::try_from(img).ok()?;
            if !img_geom.intersects(&aoi_geom).ok()? {
                return None;
            }
            let clipped = img_geom.intersection(&aoi_geom).ok()?;
            // Flatten MultiPolygon to its largest component; drop lines/points.
            let geo_clipped = GeoGeometry::<f64>::try_from(clipped).ok()?;
            match geo_clipped {
                GeoGeometry::Polygon(p) => Some(p),
                GeoGeometry::MultiPolygon(mp) => {
                    mp.0.into_iter()
                        .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap())
                }
                _ => None,
            }
        })
        .collect()
}

// ── Validation helpers ────────────────────────────────────────────────────────

/// Check that the union of all fragments covers the AOI (GEOS oracle).
fn assert_fragments_cover_aoi(fragments: &[Polygon<f64>], aoi: &Polygon<f64>, label: &str) {
    let aoi_geom = GeosGeometry::try_from(aoi).unwrap();
    let frag_geoms: Vec<GeosGeometry> = fragments
        .iter()
        .map(|p| GeosGeometry::try_from(p).unwrap())
        .collect();

    let union = GeosGeometry::create_geometry_collection(frag_geoms)
        .unwrap()
        .unary_union()
        .unwrap();

    // Small buffer to absorb floating-point boundary gaps.
    let union_buffered = union.buffer(1e-8, 3).unwrap();
    assert!(
        union_buffered.covers(&aoi_geom).unwrap(),
        "{label}: fragment union does not cover AOI"
    );
    println!("  ✓ {label}: {n} fragments cover AOI", n = fragments.len());
}

/// Check that for every image, the union of its assigned fragments matches
/// the (clipped) image polygon — symmetric difference area < tolerance.
fn assert_image_fragments_match(
    clipped_images: &[Polygon<f64>],
    fragments: &[Polygon<f64>],
    images_to_fragments: &[Vec<usize>],
    label: &str,
) {
    // Relative tolerance: sym-diff area < 0.1% of image area.
    const REL_TOL: f64 = 0.001;

    let mut max_rel_err: f64 = 0.0;

    for (img_idx, frag_indices) in images_to_fragments.iter().enumerate() {
        assert!(
            !frag_indices.is_empty(),
            "{label}: image {img_idx} has no fragments assigned"
        );

        let img_geom = GeosGeometry::try_from(&clipped_images[img_idx]).unwrap();
        let img_area = img_geom.area().unwrap();

        // Union of assigned fragments (GEOS oracle), then symmetric difference.
        let assigned: Vec<GeosGeometry> = frag_indices
            .iter()
            .map(|&fi| GeosGeometry::try_from(&fragments[fi]).unwrap())
            .collect();
        let frag_union = GeosGeometry::create_geometry_collection(assigned)
            .unwrap()
            .unary_union()
            .unwrap();

        let sym_diff_area = img_geom
            .sym_difference(&frag_union)
            .unwrap()
            .area()
            .unwrap();
        let rel_err = sym_diff_area / img_area;
        max_rel_err = max_rel_err.max(rel_err);

        assert!(
            rel_err < REL_TOL,
            "{label}: image {img_idx} sym-diff area = {sym_diff_area:.2e} ({:.4}% of image area)",
            rel_err * 100.0
        );
    }

    println!(
        "  ✓ {label}: all {} images match their fragments (max rel err = {:.2e})",
        clipped_images.len(),
        max_rel_err
    );
}

/// The invariant that makes the discrete problem sound: EVERY fragment must be
/// covered by at least one image. A fragment covered by zero images ("hole")
/// means the set-cover universe contains an element no image can satisfy — the
/// instance is then infeasible, or (if such fragments are silently dropped) a
/// "feasible" selection can leave an AOI gap uncovered. Either way the
/// feasible ⟺ covers-AOI equivalence breaks.
fn assert_no_zero_cover_fragments(
    fragments: &[Polygon<f64>],
    images_to_fragments: &[Vec<usize>],
    aoi: &Polygon<f64>,
    label: &str,
) {
    let mut covered = vec![false; fragments.len()];
    for frags in images_to_fragments {
        for &f in frags {
            covered[f] = true;
        }
    }
    let holes: Vec<usize> = (0..fragments.len()).filter(|&i| !covered[i]).collect();

    if !holes.is_empty() {
        let aoi_area = aoi.unsigned_area();
        let hole_area: f64 = holes.iter().map(|&i| fragments[i].unsigned_area()).sum();
        panic!(
            "{label}: {} of {} fragments are covered by ZERO images \
             (area {:.2} m² = {:.4}% of AOI). Discrete problem is unsound. \
             Example fragment indices: {:?}",
            holes.len(),
            fragments.len(),
            hole_area,
            hole_area / aoi_area * 100.0,
            &holes[..holes.len().min(8)],
        );
    }
    println!("  ✓ {label}: every fragment covered by ≥1 image (universe sound)");
}

// ── Test cases ────────────────────────────────────────────────────────────────

fn run_validation(city: &str, n_images: usize) {
    let aoi_file = aoi_path(city);
    let img_file = images_path(city, n_images);
    let images = load_polygons(&img_file);
    assert_eq!(images.len(), n_images, "loaded wrong number of images");
    run_validation_paths(&aoi_file, &img_file, &format!("{city}/{n_images}"));
}

/// Path-based validation so both the curated publication sets and the freshly
/// fetched biobjective pools (different directory layout) can be exercised.
fn run_validation_paths(aoi_file: &str, img_file: &str, label: &str) {
    println!("\n[{label}]");

    let aoi = load_aoi(aoi_file);
    let images = load_polygons(img_file);

    let clipped = clip_to_aoi(&images, &aoi);
    println!(
        "  clipped: {}/{} images intersect AOI",
        clipped.len(),
        images.len()
    );
    assert!(!clipped.is_empty(), "no images intersect AOI");

    let Fragmentation {
        fragments,
        images_to_fragments,
    } = fragmentize(&clipped);
    println!("  fragments: {}", fragments.len());

    assert_fragments_cover_aoi(&fragments, &aoi, label);
    assert_no_zero_cover_fragments(&fragments, &images_to_fragments, &aoi, label);
    assert_image_fragments_match(&clipped, &fragments, &images_to_fragments, label);
}

/// Same invariants, but exercising the GEOS backend directly (in addition to
/// the default pure-Rust one) as a further cross-check of the arrangement.
fn run_validation_geos(city: &str, n_images: usize) {
    let label = format!("{city}/{n_images}-geos");
    println!("\n[{label}]");

    let aoi = load_aoi(&aoi_path(city));
    let images = load_polygons(&images_path(city, n_images));
    let clipped = clip_to_aoi(&images, &aoi);
    assert!(!clipped.is_empty(), "no images intersect AOI");

    let (fragments, images_to_fragments) =
        sims_preprocessing::fragmentize_geos::fragmentize(&clipped).expect("geos fragmentize");
    println!("  fragments: {}", fragments.len());

    assert_fragments_cover_aoi(&fragments, &aoi, &label);
    assert_no_zero_cover_fragments(&fragments, &images_to_fragments, &aoi, &label);
    assert_image_fragments_match(&clipped, &fragments, &images_to_fragments, &label);
}

#[test]
fn validate_geos_paris_100() {
    run_validation_geos("paris", 100);
}

#[test]
fn validate_geos_mexico_city_100() {
    run_validation_geos("mexico_city", 100);
}

#[test]
fn validate_geos_lagos_200() {
    run_validation_geos("lagos_nigeria", 200);
}

/// Aligned (raw image, GEOS-clipped image) pairs, dropping images whose
/// intersection with the AOI is empty or degenerate. GEOS is used only to build
/// an index-aligned oracle for the atomicity check — the fragmentizer under test
/// receives the raw (unmodified) images, so its path stays fully pure-Rust.
fn clip_pairs(
    images: &[Polygon<f64>],
    aoi: &Polygon<f64>,
) -> (Vec<Polygon<f64>>, Vec<Polygon<f64>>) {
    let aoi_geom = GeosGeometry::try_from(aoi).expect("aoi to geos");
    let mut raw = Vec::new();
    let mut clipped = Vec::new();
    for img in images {
        let Ok(img_geom) = GeosGeometry::try_from(img) else {
            continue;
        };
        let Ok(inter) = img_geom.intersection(&aoi_geom) else {
            continue;
        };
        let Ok(geo_clip) = GeoGeometry::<f64>::try_from(inter) else {
            continue;
        };
        let poly = match geo_clip {
            GeoGeometry::Polygon(p) => Some(p),
            GeoGeometry::MultiPolygon(mp) => {
                mp.0.into_iter()
                    .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap())
            }
            _ => None,
        };
        if let Some(p) = poly {
            raw.push(img.clone());
            clipped.push(p);
        }
    }
    (raw, clipped)
}

/// Validate the single-pass `fragmentize_within_aoi` on a fully pure-Rust path
/// (no GEOS pre-clip — the raw images are clipped *and* fragmentized in one
/// i_overlay pass), against the GEOS oracle. This is the configuration that
/// previously failed with a separate geo clip; it should now be sound.
fn run_validation_folded_paths(aoi_file: &str, img_file: &str, label: &str) {
    use sims_preprocessing::fragmentize::fragmentize_within_aoi;
    println!("\n[{label}]");

    let aoi = load_aoi(aoi_file);
    let raw = load_polygons(img_file);
    // Index-aligned oracle clips; the fragmentizer still gets the raw images.
    let (images, clipped) = clip_pairs(&raw, &aoi);
    assert!(!images.is_empty(), "no images intersect AOI");

    let (fragments, images_to_fragments) = fragmentize_within_aoi(&images, &aoi);
    println!("  fragments: {}", fragments.len());

    assert_fragments_cover_aoi(&fragments, &aoi, label);
    assert_no_zero_cover_fragments(&fragments, &images_to_fragments, &aoi, label);
    assert_image_fragments_match(&clipped, &fragments, &images_to_fragments, label);
}

fn run_validation_folded(city: &str, n_images: usize) {
    run_validation_folded_paths(
        &aoi_path(city),
        &images_path(city, n_images),
        &format!("{city}/{n_images}-folded"),
    );
}

/// Single-pass folded validation on the large biobjective pools. Uses fast
/// pure-Rust checks only (area-partition + zero-cover) — the GEOS oracle's
/// `unary_union` over 1M+ fragments is impractical at this scale, and these are
/// the invariants that matter for detecting non-full-coverage instances.
fn run_biobjective_folded(city: &str) {
    use sims_preprocessing::fragmentize::fragmentize_within_aoi;
    let (aoi_file, img_file) = biobjective_paths(city);
    let label = format!("biobjective/{city}-folded");
    println!("\n[{label}]");

    let aoi = load_aoi(&aoi_file);
    let images = load_polygons(&img_file);

    let t = std::time::Instant::now();
    let (fragments, images_to_fragments) = fragmentize_within_aoi(&images, &aoi);
    println!(
        "  {} images → {} fragments in {:.1}s",
        images.len(),
        fragments.len(),
        t.elapsed().as_secs_f64()
    );

    // Fragments partition the AOI: their areas sum to the AOI area.
    let frag_area: f64 = fragments.iter().map(|f| f.unsigned_area()).sum();
    let aoi_area = aoi.unsigned_area();
    let rel = (frag_area - aoi_area).abs() / aoi_area;
    assert!(
        rel < 1e-4,
        "{label}: fragment areas sum to {frag_area:.3e} vs AOI {aoi_area:.3e} ({:.4}% off)",
        rel * 100.0
    );
    println!("  ✓ {label}: fragments partition the AOI (area match)");

    // Every fragment covered by ≥1 image — surfaces coverage holes (rio).
    assert_no_zero_cover_fragments(&fragments, &images_to_fragments, &aoi, &label);
}

#[test]
fn validate_folded_paris_100() {
    run_validation_folded("paris", 100);
}

#[test]
fn validate_folded_mexico_city_100() {
    run_validation_folded("mexico_city", 100);
}

#[test]
fn validate_folded_lagos_200() {
    run_validation_folded("lagos_nigeria", 200);
}

/// Stress the freshly fetched biobjective pools (phr+pneo, SPOT-excluded,
/// slivers trimmed). Ignored by default because they are large (1000–1700
/// images → tens of thousands of fragments). Run with `--ignored`.
fn biobjective_paths(city: &str) -> (String, String) {
    let aoi = format!("../sims-core/src/sims/core/assets/{city}_300/geodata/aoi.geojson");
    let dir = "../publication-data/satellite-data/biobjective_fetch";
    let img = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name().and_then(|s| s.to_str()).is_some_and(|s| {
                s.starts_with(&format!("{city}_")) && s.ends_with("_image_set.geojson")
            })
        })
        .unwrap_or_else(|| panic!("no biobjective image set for {city} in {dir}"));
    (aoi, img.to_str().unwrap().to_string())
}

fn run_biobjective(city: &str) {
    let (aoi, img) = biobjective_paths(city);
    run_validation_paths(&aoi, &img, &format!("biobjective/{city}"));
}

/// KNOWN-DEFECTIVE DATA (not a code bug): the rio_de_janeiro biobjective pool
/// leaves a ~84 km² (2.16% of AOI) hole — two coastal/ocean blobs in the south
/// of the AOI that only the excluded SPOT swaths covered. This test is EXPECTED
/// to fail the coverage invariant; it demonstrates that the preprocessing
/// correctly rejects a non-full-coverage instance. Re-fetch rio including SPOT
/// (or without the <2% sliver trim over that region) to make it a valid SIMS
/// instance.
#[test]
#[ignore = "large + known-defective data: rio pool has a 2.16% coverage hole"]
fn validate_biobjective_rio() {
    run_biobjective("rio_de_janeiro");
}

#[test]
#[ignore = "large: run explicitly with --ignored"]
fn validate_biobjective_mexico() {
    run_biobjective("mexico_city");
}

/// Single-pass fold on a large, valid biobjective pool (tokyo, ~1749 images →
/// ~1.6M fragments). All invariants must hold on the fully pure-Rust path.
#[test]
#[ignore = "large: run explicitly with --ignored"]
fn validate_biobjective_folded_tokyo() {
    run_biobjective_folded("tokyo_bay");
}

/// Single-pass fold on the KNOWN-DEFECTIVE rio pool (~2.16% coverage hole). The
/// fold must SURFACE the hole as fragments covered by zero images rather than
/// hide it — so `assert_no_zero_cover_fragments` is expected to panic. This
/// verifies the fold detects non-full-coverage instances at scale.
#[test]
#[ignore = "large + known-defective data: rio pool has a 2.16% coverage hole"]
#[should_panic(expected = "covered by ZERO images")]
fn validate_biobjective_folded_rio() {
    run_biobjective_folded("rio_de_janeiro");
}

#[test]
fn validate_paris_100() {
    run_validation("paris", 100);
}

#[test]
fn validate_lagos_100() {
    run_validation("lagos_nigeria", 100);
}

#[test]
fn validate_mexico_city_100() {
    run_validation("mexico_city", 100);
}

#[test]
fn validate_rio_100() {
    run_validation("rio_de_janeiro", 100);
}

#[test]
fn validate_tokyo_100() {
    run_validation("tokyo_bay", 100);
}

#[test]
fn validate_paris_200() {
    run_validation("paris", 200);
}

#[test]
fn validate_lagos_200() {
    run_validation("lagos_nigeria", 200);
}
