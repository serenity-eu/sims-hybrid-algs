//! Fold an instance's OCM cloud contours into its fragmentation and emit a
//! solver-ready decomposition (fragment areas + image/cloud fragment sets).
//!
//! Reads image polygons (with `id`) and the AOI from GeoJSON, and each image's
//! simplified cloud MultiPolygon from `<clouds_dir>/<id>.geojson`, then runs
//! [`fragmentize_within_aoi_with_clouds`]. Coordinates are used as-is, so feed
//! a planar/projected CRS (e.g. UTM) if you want fragment areas in m².
//!
//! Usage:
//!   fold_instance <images.geojson> <aoi.geojson> <clouds_dir> <out.json>
//! Emits JSON: {num_images, universe, areas, images_to_fragments,
//!              images_to_cloudy_fragments}; prints counts + timing to stderr.

use geo::{Area, LineString, MultiPolygon, Polygon, Simplify};
use sims_preprocessing::fragmentize::fragmentize_within_aoi_with_clouds;
use std::path::Path;

/// Coarsen a cloud MultiPolygon before folding: drop polygons below `min_area`
/// (specks) and RDP-simplify the rest at `simp` (both in the input CRS units).
fn coarsen(mp: MultiPolygon<f64>, simp: f64, min_area: f64) -> MultiPolygon<f64> {
    let kept: Vec<Polygon<f64>> =
        mp.0.into_iter()
            .filter(|p| p.unsigned_area() >= min_area)
            .collect();
    let mp = MultiPolygon::new(kept);
    if simp > 0.0 { mp.simplify(simp) } else { mp }
}

fn ring(v: &serde_json::Value) -> Option<LineString<f64>> {
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

fn poly(v: &serde_json::Value) -> Option<Polygon<f64>> {
    let rings = v.as_array()?;
    Some(Polygon::new(
        ring(rings.first()?)?,
        rings[1..].iter().filter_map(ring).collect(),
    ))
}

fn largest_poly(geom: &serde_json::Value) -> Option<Polygon<f64>> {
    match geom["type"].as_str()? {
        "Polygon" => poly(&geom["coordinates"]),
        "MultiPolygon" => geom["coordinates"]
            .as_array()?
            .iter()
            .filter_map(poly)
            .max_by(|a, b| a.unsigned_area().partial_cmp(&b.unsigned_area()).unwrap()),
        _ => None,
    }
}

fn multipoly(geom: &serde_json::Value) -> MultiPolygon<f64> {
    match geom["type"].as_str() {
        Some("MultiPolygon") => MultiPolygon::new(
            geom["coordinates"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(poly)
                .collect(),
        ),
        Some("Polygon") => MultiPolygon::new(poly(&geom["coordinates"]).into_iter().collect()),
        _ => MultiPolygon::new(vec![]),
    }
}

fn read_json(path: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (images_p, aoi_p, clouds_dir, out_p) = (&a[0], &a[1], &a[2], &a[3]);
    // Optional coarsening (input-CRS units): extra RDP tolerance + speck area floor.
    let simp: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let min_area: f64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let img_fc = read_json(images_p);
    let feats = img_fc["features"].as_array().unwrap();
    let mut images = Vec::new();
    let mut clouds = Vec::new();
    let mut image_ids = Vec::new();
    for f in feats {
        let Some(p) = largest_poly(&f["geometry"]) else {
            continue;
        };
        let id = f["properties"]["id"].as_str().unwrap_or("");
        image_ids.push(id.to_string());
        let cpath = format!("{clouds_dir}/{id}.geojson");
        let cloud = if Path::new(&cpath).exists() {
            let cfc = read_json(&cpath);
            cfc["features"]
                .as_array()
                .and_then(|fs| fs.first())
                .map(|f0| multipoly(&f0["geometry"]))
                .unwrap_or_else(|| MultiPolygon::new(vec![]))
        } else {
            MultiPolygon::new(vec![])
        };
        let cloud = coarsen(cloud, simp, min_area);
        images.push(p);
        clouds.push(cloud);
    }

    let aoi = largest_poly(&read_json(aoi_p)["features"][0]["geometry"]).expect("aoi polygon");

    let t = std::time::Instant::now();
    let cf = fragmentize_within_aoi_with_clouds(&images, &aoi, &clouds);
    let dt = t.elapsed().as_secs_f64();

    let areas: Vec<f64> = cf.fragments.iter().map(|p| p.unsigned_area()).collect();
    let assign: usize = cf.images_to_fragments.iter().map(|v| v.len()).sum();
    let cloudy: usize = cf.images_to_cloudy_fragments.iter().map(|v| v.len()).sum();
    eprintln!(
        "images={} fragments={} assignments={} cloudy_assignments={} fold={:.1}s",
        images.len(),
        cf.fragments.len(),
        assign,
        cloudy,
        dt
    );

    let doc = serde_json::json!({
        "num_images": images.len(),
        "universe": cf.fragments.len(),
        "image_ids": image_ids,
        "areas": areas,
        "images_to_fragments": cf.images_to_fragments,
        "images_to_cloudy_fragments": cf.images_to_cloudy_fragments,
    });
    std::fs::write(out_p, serde_json::to_string(&doc).unwrap()).unwrap();
}
