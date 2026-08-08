//! Benchmark harness for `fragmentize`.
//!
//! Usage: cargo run --release --example bench_fragmentize -- <city> <n>
//! e.g.   cargo run --release --example bench_fragmentize -- mexico_city 250
//!
//! With `--features geos` the GEOS backend is also run and the two are compared.

use geo::{Area, BooleanOps, LineString, Polygon};
use sims_preprocessing::fragmentize::{Fragmentation, fragmentize};
use std::time::Instant;

const DATA_ROOT: &str = "../publication-data/satellite-data";

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

fn load_aoi(path: &str) -> Polygon<f64> {
    load_polygons(path)
        .into_iter()
        .next()
        .expect("no polygon in aoi")
}

fn clip_to_aoi(images: &[Polygon<f64>], aoi: &Polygon<f64>) -> Vec<Polygon<f64>> {
    images
        .iter()
        .filter_map(|img| {
            // Intersect (empty if disjoint); keep the largest component.
            img.intersection(aoi)
                .0
                .into_iter()
                .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap())
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_else(|| "mexico_city".into());

    // "bio <city>" runs the dense biobjective pool; otherwise "<city> <n>".
    let (aoi_file, img_file, n): (String, String, String) = if first == "bio" {
        let city = args.next().expect("bio needs a city");
        let aoi = format!("../sims-core/src/sims/core/assets/{city}_300/geodata/aoi.geojson");
        let dir = format!("{DATA_ROOT}/biobjective_fetch");
        let img = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name().and_then(|s| s.to_str()).is_some_and(|s| {
                    s.starts_with(&format!("{city}_")) && s.ends_with("_image_set.geojson")
                })
            })
            .expect("no biobjective image set");
        (
            aoi,
            img.to_str().unwrap().to_string(),
            format!("bio-{city}"),
        )
    } else {
        let city = first;
        let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(250);
        (
            format!("{DATA_ROOT}/aois/{city}.geojson"),
            format!("{DATA_ROOT}/images/{city}_{n}_images.geojson"),
            n.to_string(),
        )
    };
    let city = "";

    let t0 = Instant::now();
    let aoi = load_aoi(&aoi_file);
    let images = load_polygons(&img_file);
    let clipped = clip_to_aoi(&images, &aoi);
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "[{city}/{n}] load+clip: {load_ms:.1} ms  ({} images -> {} clipped)",
        images.len(),
        clipped.len()
    );

    // Optional GEOS reference backend (feature `geos`), run first for comparison.
    #[cfg(feature = "geos")]
    let geos_s = {
        let t = Instant::now();
        let (frags, i2f) =
            sims_preprocessing::fragmentize_geos::fragmentize(&clipped).expect("geos failed");
        let s = t.elapsed().as_secs_f64();
        let assign: usize = i2f.iter().map(|v| v.len()).sum();
        println!(
            "[{city}/{n}] geos      : {s:7.3} s  ({} fragments, {} assignments)",
            frags.len(),
            assign
        );
        s
    };

    // Default pure-Rust geo/i_overlay backend.
    let t = Instant::now();
    let Fragmentation {
        fragments: frags,
        images_to_fragments: i2f,
    } = fragmentize(&clipped);
    let s = t.elapsed().as_secs_f64();
    let assign: usize = i2f.iter().map(|v| v.len()).sum();
    print!(
        "[{city}/{n}] geo(iovl) : {s:7.3} s  ({} fragments, {} assignments)",
        frags.len(),
        assign
    );
    #[cfg(feature = "geos")]
    print!("  [{:.2}x vs geos]", geos_s / s);
    println!();
}
