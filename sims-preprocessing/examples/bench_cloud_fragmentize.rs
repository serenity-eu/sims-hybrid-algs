//! Benchmark: how cloud-smoothing (RDP) tolerance impacts cloud-folded
//! fragmentation performance.
//!
//! The independent variable is the `simplify_polygons` tolerance applied to each
//! image's OCM cloud MultiPolygon before it is folded into the arrangement by
//! [`fragmentize_within_aoi_with_clouds`]. Coarser tolerances collapse cloud
//! outlines to fewer vertices → fewer cutting edges → fewer fragments and a
//! faster arrangement, at some cost in cloud-boundary fidelity. This sweeps a
//! range of tolerances and reports, per tolerance, the total cloud vertex count,
//! the fragmentation wall-clock time, the resulting fragment count, and the
//! number of cloudy fragment assignments — against the cloud-free baseline
//! ([`fragmentize_within_aoi`]).
//!
//! OCM inference is run once up front (it does not depend on tolerance) and the
//! raw cloud polygons are cached; only simplify + fragmentize are re-timed.
//!
//! Usage:
//!   cargo run --release --features ocm --example bench_cloud_fragmentize -- \
//!       <city> [limit]
//!   e.g. cargo run --release --features ocm --example bench_cloud_fragmentize -- \
//!       lagos_nigeria 200

use geo::{Area, BooleanOps, BoundingRect, LineString, MultiPolygon, Polygon};
use sims_preprocessing::cloud::{cloud_multipolygon, is_cloudy, simplify_polygons, vertex_count};
use sims_preprocessing::cloud_ocm::{OcmEnsemble, label_raster_from_quicklook};
use sims_preprocessing::fragmentize::{fragmentize_within_aoi, fragmentize_within_aoi_with_clouds};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DATA_ROOT: &str = "../publication-data/satellite-data/biobjective_fetch";
const ASSETS_ROOT: &str = "../sims-core/src/sims/core/assets";
const TOLERANCES_PX: &[f64] = &[0.0, 0.5, 1.0, 2.0, 4.0, 8.0];

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

fn polygon_from_coords(coords: &serde_json::Value) -> Option<Polygon<f64>> {
    let rings = coords.as_array()?;
    let exterior = ring_from_coords(rings.first()?)?;
    let interiors: Vec<LineString<f64>> = rings[1..].iter().filter_map(ring_from_coords).collect();
    Some(Polygon::new(exterior, interiors))
}

/// The single (largest) polygon of a feature's geometry.
fn feature_polygon(geom: &serde_json::Value) -> Option<Polygon<f64>> {
    match geom["type"].as_str()? {
        "Polygon" => polygon_from_coords(&geom["coordinates"]),
        "MultiPolygon" => geom["coordinates"]
            .as_array()?
            .iter()
            .filter_map(polygon_from_coords)
            .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap()),
        _ => None,
    }
}

fn load_aoi(path: &str) -> Polygon<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let fc: serde_json::Value = serde_json::from_str(&text).unwrap();
    feature_polygon(&fc["features"][0]["geometry"]).expect("no polygon in aoi")
}

/// Locate the `<city>_<n>_image_set.geojson` for a city.
fn image_set_path(city: &str) -> PathBuf {
    std::fs::read_dir(DATA_ROOT)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name().and_then(|s| s.to_str()).is_some_and(|s| {
                s.starts_with(&format!("{city}_")) && s.ends_with("_image_set.geojson")
            })
        })
        .unwrap_or_else(|| panic!("no image set for {city} under {DATA_ROOT}"))
}

/// `(id, footprint)` for every feature.
fn load_image_set(path: &Path) -> Vec<(String, Polygon<f64>)> {
    let text = std::fs::read_to_string(path).unwrap();
    let fc: serde_json::Value = serde_json::from_str(&text).unwrap();
    fc["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| {
            let id = f["properties"]["id"].as_str()?.to_string();
            let poly = feature_polygon(&f["geometry"])?;
            Some((id, poly))
        })
        .collect()
}

fn bbox_of(poly: &Polygon<f64>) -> [f64; 4] {
    let r = poly.bounding_rect().unwrap();
    [r.min().x, r.min().y, r.max().x, r.max().y]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let city = args.next().unwrap_or_else(|| "lagos_nigeria".into());
    let limit: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let aoi = load_aoi(&format!("{ASSETS_ROOT}/{city}_300/geodata/aoi.geojson"));
    let image_set = load_image_set(&image_set_path(city.as_str()));
    let qdir = format!("{DATA_ROOT}/quicklooks/{city}");

    // Keep images that (a) have a quicklook and (b) intersect the AOI. Clip the
    // footprint to the AOI so the pool matches what fragmentation actually sees.
    let mut kept: Vec<(PathBuf, [f64; 4], Polygon<f64>)> = Vec::new();
    for (id, footprint) in &image_set {
        let png = PathBuf::from(format!("{qdir}/{id}.png"));
        if !png.exists() {
            continue;
        }
        let Some(clipped) = footprint
            .intersection(&aoi)
            .0
            .into_iter()
            .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap())
        else {
            continue;
        };
        kept.push((png, bbox_of(footprint), clipped));
        if kept.len() >= limit {
            break;
        }
    }
    assert!(!kept.is_empty(), "no quicklook images intersect the AOI");
    println!(
        "[{city}] {} images with quicklooks intersect AOI (limit {})",
        kept.len(),
        if limit == usize::MAX {
            "none".into()
        } else {
            limit.to_string()
        }
    );

    // ── OCM inference (one-time; tolerance-independent) ──────────────────────
    let mut model =
        OcmEnsemble::load(Path::new(&format!("{DATA_ROOT}/models_onnx"))).expect("load OCM model");
    let clipped: Vec<Polygon<f64>> = kept.iter().map(|(_, _, c)| c.clone()).collect();
    // raw cloud MultiPolygon (cloud+shadow) and the pixel size in degrees per image.
    let mut raw_clouds: Vec<MultiPolygon<f64>> = Vec::with_capacity(kept.len());
    let mut px_deg: Vec<f64> = Vec::with_capacity(kept.len());
    let t_ocm = Instant::now();
    for (png, bbox, _) in &kept {
        let raster = label_raster_from_quicklook(&mut model, png, *bbox).expect("ocm inference");
        px_deg.push((bbox[2] - bbox[0]) / raster.width.max(1) as f64);
        raw_clouds.push(cloud_multipolygon(&raster, is_cloudy));
    }
    let ocm_s = t_ocm.elapsed().as_secs_f64();
    let cloudy_imgs = raw_clouds.iter().filter(|mp| !mp.0.is_empty()).count();
    println!(
        "OCM inference: {ocm_s:.2}s for {} images ({cloudy_imgs} have cloud/shadow), {:.1} ms/img\n",
        kept.len(),
        ocm_s / kept.len() as f64 * 1e3
    );

    // ── cloud-free baseline ──────────────────────────────────────────────────
    let t = Instant::now();
    let base = fragmentize_within_aoi(&clipped, &aoi);
    let base_s = t.elapsed().as_secs_f64();
    println!(
        "{:<10} {:>12} {:>10} {:>12} {:>12} {:>12}",
        "tol(px)", "cloud_verts", "frag_ms", "fragments", "cloudy_frag", "vs_base"
    );
    println!(
        "{:<10} {:>12} {:>10.1} {:>12} {:>12} {:>12}",
        "none",
        0,
        base_s * 1e3,
        base.fragments.len(),
        0,
        "1.00x"
    );

    // ── tolerance sweep ──────────────────────────────────────────────────────
    // Silence panic backtraces from the caught i_overlay solver panics below so
    // the table stays readable; only the sweep runs past this point.
    std::panic::set_hook(Box::new(|_| {}));
    for &tol_px in TOLERANCES_PX {
        // Simplify each image's cloud with its own px→deg scale.
        let clouds: Vec<MultiPolygon<f64>> = raw_clouds
            .iter()
            .zip(&px_deg)
            .map(|(mp, &pd)| simplify_polygons(mp, tol_px * pd))
            .collect();
        let cloud_verts: usize = clouds.iter().map(vertex_count).sum();
        let label = if tol_px == 0.0 {
            "0(raw)".to_string()
        } else {
            format!("{tol_px}")
        };

        // The dense, near-degenerate raw marching-squares outlines can trip an
        // i_overlay solver panic; catch it so the sweep reports the first stable
        // tolerance rather than aborting.
        let t = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fragmentize_within_aoi_with_clouds(&clipped, &aoi, &clouds)
        }));
        let frag_s = t.elapsed().as_secs_f64();
        match result {
            Ok(cf) => {
                let cloudy: usize = cf.images_to_cloudy_fragments.iter().map(|v| v.len()).sum();
                println!(
                    "{:<10} {:>12} {:>10.1} {:>12} {:>12} {:>11.2}x",
                    label,
                    cloud_verts,
                    frag_s * 1e3,
                    cf.fragments.len(),
                    cloudy,
                    frag_s / base_s
                );
            }
            Err(_) => {
                println!(
                    "{:<10} {:>12} {:>10} {:>12} {:>12} {:>12}",
                    label, cloud_verts, "PANIC", "-", "-", "-"
                );
            }
        }
    }
}
