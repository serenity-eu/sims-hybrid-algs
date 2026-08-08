//! Batch cloud masking for a city's instance quicklooks.
//!
//! For every pool image that has a quicklook, runs the OCM ensemble to a
//! `LabelRaster` (the mask), vectorizes the cloud+shadow region to a
//! MultiPolygon (contours), simplifies it (RDP), and writes the georeferenced
//! result as GeoJSON. Also writes a per-city `summary.csv`.
//!
//! Usage:
//!   cargo run --release --features ocm --example mask_instances -- <city> [simplify_px=2] [limit]
//!
//! Paths are relative to the `sims-preprocessing/` crate dir.

use geo::{Area, BoundingRect, LineString, MultiPolygon, Polygon};
use sims_preprocessing::cloud::{
    NODATA, cloud_multipolygon, is_cloudy, simplify_polygons, vertex_count,
};
use sims_preprocessing::cloud_ocm::{OcmEnsemble, label_raster_from_quicklook};
use std::path::{Path, PathBuf};

const DATA: &str = "../publication-data/satellite-data";

fn ring_coords(v: &serde_json::Value) -> Option<LineString<f64>> {
    Some(LineString::from(
        v.as_array()?
            .iter()
            .filter_map(|c| {
                let a = c.as_array()?;
                Some((a[0].as_f64()?, a[1].as_f64()?))
            })
            .collect::<Vec<_>>(),
    ))
}

fn poly_coords(v: &serde_json::Value) -> Option<Polygon<f64>> {
    let rings = v.as_array()?;
    Some(Polygon::new(
        ring_coords(rings.first()?)?,
        rings[1..].iter().filter_map(ring_coords).collect(),
    ))
}

/// Largest polygon of a feature geometry (Polygon or MultiPolygon).
fn feature_polygon(geom: &serde_json::Value) -> Option<Polygon<f64>> {
    match geom["type"].as_str()? {
        "Polygon" => poly_coords(&geom["coordinates"]),
        "MultiPolygon" => geom["coordinates"]
            .as_array()?
            .iter()
            .filter_map(poly_coords)
            .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap()),
        _ => None,
    }
}

fn multipolygon_geojson(
    id: &str,
    cloud_pct: f64,
    raw_v: usize,
    simp_v: usize,
    mp: &MultiPolygon<f64>,
) -> String {
    let ring = |ls: &LineString<f64>| {
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
        "{{\"type\":\"FeatureCollection\",\"features\":[{{\"type\":\"Feature\",\
         \"properties\":{{\"id\":\"{id}\",\"cloud_pct\":{cloud_pct:.2},\
         \"raw_vertices\":{raw_v},\"simplified_vertices\":{simp_v}}},\
         \"geometry\":{{\"type\":\"MultiPolygon\",\"coordinates\":[{}]}}}}]}}",
        polys.join(",")
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let city = args
        .first()
        .cloned()
        .unwrap_or_else(|| "lagos_nigeria".into());
    let simplify_px: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let limit: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let pool = format!("{DATA}/instances_pool/{city}_image_set.geojson");
    let qdir = PathBuf::from(format!("{DATA}/instances_quicklooks/{city}"));
    let out_dir = PathBuf::from(format!("{DATA}/instances_cloud_masks/{city}"));
    std::fs::create_dir_all(&out_dir)?;

    let fc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&pool)?)?;
    let feats = fc["features"].as_array().ok_or("no features")?;

    let mut model = OcmEnsemble::load(Path::new(&format!("{DATA}/biobjective_fetch/models_onnx")))?;

    // Preserve prior rows on incremental runs (new images are appended).
    let summary_path = out_dir.join("summary.csv");
    let mut summary = std::fs::read_to_string(&summary_path).unwrap_or_else(|_| {
        String::from("id,cloud_pct,raw_vertices,simplified_vertices,polygons\n")
    });
    let (mut done, mut skipped) = (0usize, 0usize);
    let t0 = std::time::Instant::now();

    for f in feats {
        if done >= limit {
            break;
        }
        let id = match f["properties"]["id"].as_str() {
            Some(s) => s,
            None => continue,
        };
        let png = qdir.join(format!("{id}.png"));
        if !png.exists() {
            skipped += 1;
            continue; // only images used in instances have quicklooks
        }
        if out_dir.join(format!("{id}.geojson")).exists() {
            skipped += 1;
            continue; // already masked (resumable / incremental runs)
        }
        let Some(footprint) = feature_polygon(&f["geometry"]) else {
            continue;
        };
        let rect = footprint.bounding_rect().unwrap();
        let bbox = [rect.min().x, rect.min().y, rect.max().x, rect.max().y];

        let raster = label_raster_from_quicklook(&mut model, &png, bbox)?;
        let valid = raster
            .labels
            .iter()
            .filter(|&&l| l != NODATA)
            .count()
            .max(1);
        let cloudy = raster.labels.iter().filter(|&&l| is_cloudy(l)).count();
        let cloud_pct = cloudy as f64 / valid as f64 * 100.0;

        let raw = cloud_multipolygon(&raster, is_cloudy);
        let px_deg = (bbox[2] - bbox[0]) / raster.width.max(1) as f64;
        let simp = simplify_polygons(&raw, simplify_px * px_deg);

        let (rv, sv) = (vertex_count(&raw), vertex_count(&simp));
        std::fs::write(
            out_dir.join(format!("{id}.geojson")),
            multipolygon_geojson(id, cloud_pct, rv, sv, &simp),
        )?;
        summary.push_str(&format!("{id},{cloud_pct:.2},{rv},{sv},{}\n", simp.0.len()));
        done += 1;
        if done % 50 == 0 {
            println!(
                "  {city}: {done} masked ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
        }
    }

    std::fs::write(&summary_path, summary)?;
    println!(
        "{city}: masked {done} images (skipped {skipped} without quicklook) in {:.1}s -> {}",
        t0.elapsed().as_secs_f64(),
        out_dir.display()
    );
    Ok(())
}
