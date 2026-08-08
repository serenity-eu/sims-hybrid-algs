//! Demo of the `ocm` cloud-mask inference API (not wired into the pipeline).
//!
//! Usage:
//!   cargo run --release --features ocm --example cloud_ocm_demo -- \
//!       <model_dir> <quicklook.png> <min_lon> <min_lat> <max_lon> <max_lat>
//!
//! Loads the OCM ONNX ensemble, produces a georeferenced LabelRaster, and prints
//! the cloud / shadow / cloud+shadow coverage over valid pixels.

use geo::Area;
use sims_preprocessing::cloud::{
    NODATA, cloud_multipolygon, is_cloudy, simplify_polygons, vertex_count,
};
use sims_preprocessing::cloud_ocm::{OcmEnsemble, label_raster_from_quicklook};
use std::path::Path;

/// Minimal GeoJSON FeatureCollection with the cloud MultiPolygon.
fn multipolygon_geojson(mp: &geo::MultiPolygon<f64>) -> String {
    let ring = |ls: &geo::LineString<f64>| {
        let pts: Vec<String> = ls.0.iter().map(|c| format!("[{},{}]", c.x, c.y)).collect();
        format!("[{}]", pts.join(","))
    };
    let polys: Vec<String> =
        mp.0.iter()
            .map(|p| {
                let mut rings = vec![ring(p.exterior())];
                rings.extend(p.interiors().iter().map(ring));
                format!("[{}]", rings.join(","))
            })
            .collect();
    format!(
        "{{\"type\":\"FeatureCollection\",\"features\":[{{\"type\":\"Feature\",\"properties\":{{}},\
         \"geometry\":{{\"type\":\"MultiPolygon\",\"coordinates\":[{}]}}}}]}}",
        polys.join(",")
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: <model_dir> <png> <min_lon> <min_lat> <max_lon> <max_lat>");
        std::process::exit(2);
    }
    let bbox = [a[3].parse()?, a[4].parse()?, a[5].parse()?, a[6].parse()?];

    let mut model = OcmEnsemble::load(Path::new(&a[1]))?;
    let t = std::time::Instant::now();
    let r = label_raster_from_quicklook(&mut model, Path::new(&a[2]), bbox)?;
    let dt = t.elapsed().as_secs_f64();

    let valid = r.labels.iter().filter(|&&l| l != NODATA).count().max(1);
    let cloud = r.labels.iter().filter(|&&l| l == 1).count();
    let shadow = r.labels.iter().filter(|&&l| l == 2).count();
    let cloudy = r.labels.iter().filter(|&&l| is_cloudy(l)).count();
    let pct = |n: usize| n as f64 / valid as f64 * 100.0;
    println!(
        "{}x{}  inference {:.2}s  valid={}",
        r.width, r.height, dt, valid
    );
    println!(
        "cloud={:.1}%  shadow={:.1}%  cloud+shadow={:.1}%",
        pct(cloud),
        pct(shadow),
        pct(cloudy)
    );

    // ── vectorize: one polygon per cloud (cloud-only borders) ──
    let clouds = cloud_multipolygon(&r, |l| l == 1);
    let poly_area: f64 = clouds.unsigned_area();
    // footprint area in the same (lon/lat deg²) units, for a consistency check
    let [x0, y0, x1, y1] = r.bbox;
    let footprint_area = (x1 - x0) * (y1 - y0) * valid as f64 / (r.width * r.height) as f64;
    println!(
        "cloud polygons: {}  area={:.3e} deg²  (~{:.1}% of footprint — vs {:.1}% pixels)",
        clouds.0.len(),
        poly_area,
        poly_area / footprint_area * 100.0,
        pct(cloud)
    );
    std::fs::write("clouds_raw.geojson", multipolygon_geojson(&clouds))?;

    // ── simplify (Ramer–Douglas–Peucker): arg [7] = tolerance in pixels (def 1.5) ──
    let px_deg = (x1 - x0) / r.width as f64;
    let tol_px: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let simplified = simplify_polygons(&clouds, tol_px * px_deg);
    println!(
        "vertices: {} raw -> {} simplified ({:.0}% fewer);  area {:.3e} -> {:.3e} deg²",
        vertex_count(&clouds),
        vertex_count(&simplified),
        (1.0 - vertex_count(&simplified) as f64 / vertex_count(&clouds).max(1) as f64) * 100.0,
        clouds.unsigned_area(),
        simplified.unsigned_area(),
    );
    std::fs::write("clouds.geojson", multipolygon_geojson(&simplified))?;
    println!("wrote clouds_raw.geojson + clouds.geojson (simplified)");
    Ok(())
}
