//! Dump JSON for the 3-panel cloud/fragment diagram (see `plot_cloud_diagram.py`).
//!
//! Picks up to `N` mutually **non-overlapping** images that intersect the AOI,
//! have a quicklook, and carry some cloud; runs OCM on each; folds the (2px-
//! simplified) clouds into the AOI arrangement via
//! [`fragmentize_within_aoi_with_clouds`]; and writes the AOI, the clipped image
//! footprints, the per-image cloud MultiPolygons, and the resulting fragments
//! (with per-image + cloudy assignments) to a JSON file the Python script plots.
//!
//! By default the picked images are mutually **non-overlapping** and moderately
//! cloudy (clean tiling for small diagrams). Pass `--overlap` to instead take the
//! first `n` quicklook-backed images intersecting the AOI regardless of overlap
//! or cloud fraction — the realistic dense/overlapping pool.
//!
//! Usage:
//!   cargo run --release --features ocm --example dump_cloud_diagram -- \
//!       <city> [n=4] [out.json] [--overlap]

use geo::{Area, BooleanOps, BoundingRect, LineString, MultiPolygon, Polygon};
use serde_json::{Value, json};
use sims_preprocessing::cloud::{cloud_multipolygon, is_cloudy, simplify_polygons};
use sims_preprocessing::cloud_ocm::{OcmEnsemble, label_raster_from_quicklook};
use sims_preprocessing::fragmentize::fragmentize_within_aoi_with_clouds;
use std::path::{Path, PathBuf};

const DATA_ROOT: &str = "../publication-data/satellite-data/biobjective_fetch";
const ASSETS_ROOT: &str = "../sims-core/src/sims/core/assets";
const SIMPLIFY_PX: f64 = 2.0;
const MAX_CANDIDATES: usize = 120; // cap OCM calls while searching

fn ring_from_coords(coords: &Value) -> Option<LineString<f64>> {
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

fn polygon_from_coords(coords: &Value) -> Option<Polygon<f64>> {
    let rings = coords.as_array()?;
    let exterior = ring_from_coords(rings.first()?)?;
    let interiors: Vec<LineString<f64>> = rings[1..].iter().filter_map(ring_from_coords).collect();
    Some(Polygon::new(exterior, interiors))
}

fn feature_polygon(geom: &Value) -> Option<Polygon<f64>> {
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
    let fc: Value = serde_json::from_str(&text).unwrap();
    feature_polygon(&fc["features"][0]["geometry"]).expect("no polygon in aoi")
}

fn image_set_path(city: &str) -> PathBuf {
    std::fs::read_dir(DATA_ROOT)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name().and_then(|s| s.to_str()).is_some_and(|s| {
                s.starts_with(&format!("{city}_")) && s.ends_with("_image_set.geojson")
            })
        })
        .unwrap_or_else(|| panic!("no image set for {city}"))
}

fn load_image_set(path: &Path) -> Vec<(String, Polygon<f64>)> {
    let fc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    fc["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| {
            let id = f["properties"]["id"].as_str()?.to_string();
            Some((id, feature_polygon(&f["geometry"])?))
        })
        .collect()
}

fn bbox_of(poly: &Polygon<f64>) -> [f64; 4] {
    let r = poly.bounding_rect().unwrap();
    [r.min().x, r.min().y, r.max().x, r.max().y]
}

/// Area of the overlap between two polygons.
fn overlap_area(a: &Polygon<f64>, b: &Polygon<f64>) -> f64 {
    a.intersection(b).unsigned_area()
}

/// Clip a footprint to the AOI, keeping the largest component.
fn clip(poly: &Polygon<f64>, aoi: &Polygon<f64>) -> Option<Polygon<f64>> {
    poly.intersection(aoi)
        .0
        .into_iter()
        .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap())
}

fn ring_json(ls: &LineString<f64>) -> Value {
    Value::Array(ls.0.iter().map(|c| json!([c.x, c.y])).collect())
}

fn polygon_json(p: &Polygon<f64>) -> Value {
    json!({
        "exterior": ring_json(p.exterior()),
        "holes": p.interiors().iter().map(ring_json).collect::<Vec<_>>(),
    })
}

fn multipolygon_json(mp: &MultiPolygon<f64>) -> Value {
    Value::Array(mp.0.iter().map(polygon_json).collect())
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let allow_overlap = raw.iter().any(|a| a == "--overlap");
    let mut pos = raw.iter().filter(|a| !a.starts_with("--"));
    let city = pos
        .next()
        .cloned()
        .unwrap_or_else(|| "lagos_nigeria".into());
    let target: usize = pos.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let out = pos
        .next()
        .cloned()
        .unwrap_or_else(|| format!("diagram_{city}.json"));

    let aoi = load_aoi(&format!("{ASSETS_ROOT}/{city}_300/geodata/aoi.geojson"));
    let image_set = load_image_set(&image_set_path(&city));
    let qdir = format!("{DATA_ROOT}/quicklooks/{city}");
    let mut model =
        OcmEnsemble::load(Path::new(&format!("{DATA_ROOT}/models_onnx"))).expect("load OCM");

    // Greedily pick non-overlapping, quicklook-backed, moderately-cloudy images.
    let mut sel_footprints: Vec<Polygon<f64>> = Vec::new();
    let mut sel_clipped: Vec<Polygon<f64>> = Vec::new();
    let mut sel_clouds: Vec<MultiPolygon<f64>> = Vec::new();
    let mut scanned = 0usize;

    for (id, footprint) in &image_set {
        if sel_footprints.len() >= target || scanned >= MAX_CANDIDATES {
            break;
        }
        let png = PathBuf::from(format!("{qdir}/{id}.png"));
        if !png.exists() {
            continue;
        }
        let Some(clipped) = clip(footprint, &aoi) else {
            continue;
        };
        let fp_area = footprint.unsigned_area().max(1e-12);
        // Non-overlap mode: reject if it overlaps a selected footprint by >2%.
        if !allow_overlap
            && sel_footprints
                .iter()
                .any(|s| overlap_area(footprint, s) > 0.02 * fp_area)
        {
            continue;
        }
        scanned += 1;
        let raster = label_raster_from_quicklook(&mut model, &png, bbox_of(footprint))
            .expect("ocm inference");
        let px_deg = (raster.bbox[2] - raster.bbox[0]) / raster.width.max(1) as f64;
        let cloud = simplify_polygons(
            &cloud_multipolygon(&raster, is_cloudy),
            SIMPLIFY_PX * px_deg,
        );
        let frac = cloud.unsigned_area() / fp_area;
        // Non-overlap mode also insists on a moderate cloud fraction so the small
        // diagram reads well; overlap mode takes every image as-is.
        if !allow_overlap && !(0.03..=0.85).contains(&frac) {
            continue;
        }
        if sel_footprints.len() % 10 == 0 {
            println!("  ... {} of {} selected", sel_footprints.len() + 1, target);
        }
        sel_footprints.push(footprint.clone());
        sel_clipped.push(clipped);
        sel_clouds.push(cloud);
    }
    assert!(
        !sel_clipped.is_empty(),
        "no non-overlapping cloudy images found for {city}"
    );

    let cf = fragmentize_within_aoi_with_clouds(&sel_clipped, &aoi, &sel_clouds);
    println!(
        "[{city}] {} images -> {} fragments ({} cloudy assignments)",
        sel_clipped.len(),
        cf.fragments.len(),
        cf.images_to_cloudy_fragments
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
    );

    let doc = json!({
        "city": city,
        "aoi": polygon_json(&aoi),
        "images": sel_clipped.iter().map(polygon_json).collect::<Vec<_>>(),
        "clouds": sel_clouds.iter().map(multipolygon_json).collect::<Vec<_>>(),
        "fragments": cf.fragments.iter().map(polygon_json).collect::<Vec<_>>(),
        "images_to_fragments": cf.images_to_fragments,
        "images_to_cloudy_fragments": cf.images_to_cloudy_fragments,
    });
    std::fs::write(&out, serde_json::to_string(&doc).unwrap()).unwrap();
    println!("wrote {out}");
}
