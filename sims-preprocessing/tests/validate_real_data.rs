/// Integration tests against real satellite data from publication-data/.
///
/// Two invariants are checked for each (AOI, image-set) pair:
///
/// 1. **Coverage**: the union of all fragments covers the entire AOI.
/// 2. **Image-fragment consistency**: for every image, the union of its
///    assigned fragments matches the original (clipped) image polygon,
///    i.e. their symmetric difference has negligible area.
use geo::{Geometry as GeoGeometry, Polygon};
use geos::{Geom, Geometry as GeosGeometry};
use sims_preprocessing::fragmentize::fragmentize;
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

/// Load every Polygon feature from a GeoJSON FeatureCollection.
fn load_polygons(path: &str) -> Vec<Polygon<f64>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let fc: serde_json::Value = serde_json::from_str(&text).unwrap();
    fc["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|feat| {
            let geom_str = feat["geometry"].to_string();
            let geos_geom = GeosGeometry::new_from_geojson(&geom_str).ok()?;
            let geo_geom = GeoGeometry::<f64>::try_from(geos_geom).ok()?;
            match geo_geom {
                GeoGeometry::Polygon(p) => Some(p),
                _ => None,
            }
        })
        .collect()
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
                    // keep the largest component (by bbox area)
                    mp.0.into_iter().max_by(|a, b| {
                        let area_a = a
                            .bounding_rect()
                            .map(|r| r.width() * r.height())
                            .unwrap_or(0.0);
                        let area_b = b
                            .bounding_rect()
                            .map(|r| r.width() * r.height())
                            .unwrap_or(0.0);
                        area_a.partial_cmp(&area_b).unwrap()
                    })
                }
                _ => None,
            }
        })
        .collect()
}

// ── Validation helpers ────────────────────────────────────────────────────────

/// Check that the union of all fragments covers the AOI.
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

    // Small buffer to absorb floating-point boundary gaps
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

        // Union of assigned fragments
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

// ── Test cases ────────────────────────────────────────────────────────────────

fn run_validation(city: &str, n_images: usize) {
    let aoi_file = aoi_path(city);
    let img_file = images_path(city, n_images);

    println!("\n[{city} / {n_images} images]");

    let aoi = load_aoi(&aoi_file);
    let images = load_polygons(&img_file);
    assert_eq!(images.len(), n_images, "loaded wrong number of images");

    let clipped = clip_to_aoi(&images, &aoi);
    println!(
        "  clipped: {}/{n_images} images intersect AOI",
        clipped.len()
    );
    assert!(!clipped.is_empty(), "no images intersect AOI");

    let (fragments, images_to_fragments) = fragmentize(&clipped).expect("fragmentize failed");
    println!("  fragments: {}", fragments.len());

    assert_fragments_cover_aoi(&fragments, &aoi, &format!("{city}/{n_images}"));
    assert_image_fragments_match(
        &clipped,
        &fragments,
        &images_to_fragments,
        &format!("{city}/{n_images}"),
    );
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
