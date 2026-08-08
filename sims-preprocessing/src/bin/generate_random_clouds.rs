//! Generate "random clouds" SIMS instances from a UP42 image pool.
//!
//! Pipeline (all in Rust, zero cloud-imagery processing):
//!   1. read AOI + pool (lon/lat GeoJSON),
//!   2. for each size N: coverage-aware sample of N images ([`sampler`]),
//!   3. fragmentize the selected images within the AOI ([`fragmentize_within_aoi`]),
//!   4. discretize with **random** synthetic clouds matching each image's cloud %
//!      ([`SimsProblem::discretize`] → [`clouds::generate_clouds`]),
//!   5. write `<out>/<name>_<N>.dzn` (+ a `<name>_<N>.meta.json`).
//!
//! The cloud-coverage cap is intentionally NOT applied — cloudy images are kept.
//!
//! Usage:
//!   generate_random_clouds <pool.geojson> <aoi.geojson> <out_dir> <name> \
//!       [--sizes 100,150,200,250,300,350,400,450,500] [--seed <u64>] \
//!       [--grid-long 55] [--keep-frac 0.6]

use geo::{Area, Centroid, LineString, MultiPolygon, Polygon};
use sims_preprocessing::fragmentize::fragmentize_within_aoi;
use sims_preprocessing::problem::SimsProblem;
use sims_preprocessing::sampler::{SampleParams, sample_instance};
use std::path::Path;

// ── GeoJSON helpers ──────────────────────────────────────────────────────────

fn read_json(path: &str) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path}: {e}"))
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

/// Largest polygon of a feature geometry (Polygon or MultiPolygon).
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

/// `providerProperties` may be a nested object or a JSON-encoded string.
fn provider_props(p: &serde_json::Value) -> serde_json::Value {
    match &p["providerProperties"] {
        serde_json::Value::String(s) => serde_json::from_str(s).unwrap_or(serde_json::Value::Null),
        v => v.clone(),
    }
}

// ── Per-image pool record ────────────────────────────────────────────────────

struct Pool {
    polys: Vec<Polygon<f64>>,
    mps: Vec<MultiPolygon<f64>>,
    cost: Vec<f64>,
    resolution: Vec<f64>,
    cloud: Vec<f64>,
    incidence: Vec<f64>,
}

fn read_pool(path: &str) -> Pool {
    let fc = read_json(path);
    let feats = fc["features"].as_array().expect("pool: no features");
    let mut pool = Pool {
        polys: Vec::new(),
        mps: Vec::new(),
        cost: Vec::new(),
        resolution: Vec::new(),
        cloud: Vec::new(),
        incidence: Vec::new(),
    };
    for f in feats {
        let Some(pg) = largest_poly(&f["geometry"]) else {
            continue;
        };
        let p = &f["properties"];
        let pp = provider_props(p);
        pool.cost.push(p["cost"].as_f64().unwrap_or(0.0));
        pool.resolution
            .push(p["resolution"].as_f64().unwrap_or(0.0));
        pool.cloud.push(
            p["cloudCoverage"]
                .as_f64()
                .or_else(|| p["cloud_coverage"].as_f64())
                .unwrap_or(0.0),
        );
        pool.incidence
            .push(pp["incidenceAngle"].as_f64().unwrap_or(0.0));
        pool.mps.push(MultiPolygon::new(vec![pg.clone()]));
        pool.polys.push(pg);
    }
    pool
}

// ── Reproducible per-(name, size) seed ───────────────────────────────────────

fn seed_for(name: &str, n: usize, base: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ base;
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h = (h ^ n as u64).wrapping_mul(0x0000_0100_0000_01b3);
    // SplitMix64 finaliser
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

// ── Args ─────────────────────────────────────────────────────────────────────

fn arg_val(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!(
            "usage: generate_random_clouds <pool.geojson> <aoi.geojson> <out_dir> <name> \
             [--sizes 100,150,...] [--seed N] [--grid-long 55] [--keep-frac 0.6]"
        );
        std::process::exit(2);
    }
    let (pool_p, aoi_p, out_dir, name) = (&args[0], &args[1], &args[2], &args[3]);
    let sizes: Vec<usize> = arg_val(&args, "--sizes")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| (100..=500).step_by(50).collect());
    let base_seed: u64 = arg_val(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut params = SampleParams::default();
    if let Some(g) = arg_val(&args, "--grid-long").and_then(|s| s.parse().ok()) {
        params.grid_long = g;
    }
    if let Some(k) = arg_val(&args, "--keep-frac").and_then(|s| s.parse().ok()) {
        params.keep_frac = k;
    }

    let aoi = largest_poly(&read_json(aoi_p)["features"][0]["geometry"]).expect("aoi polygon");
    // Area scale deg² → m² at the AOI centroid latitude (cylindrical-equal-area).
    let center_lat = aoi.centroid().map_or(0.0, |c| c.y());
    let k_area = 111_320.0_f64.powi(2) * center_lat.to_radians().cos();

    let pool = read_pool(pool_p);
    let n_pool = pool.polys.len();
    std::fs::create_dir_all(out_dir).unwrap();
    println!(
        "{name}: pool={n_pool} imgs, cloud mean={:.1}% (>50%: {} imgs), AOI centre lat={center_lat:.3}",
        pool.cloud.iter().sum::<f64>() / n_pool.max(1) as f64,
        pool.cloud.iter().filter(|&&c| c > 50.0).count()
    );
    println!(
        "{:>5} {:>9} {:>9} {:>10} {:>8} {:>8}",
        "N", "universe", "backbone", "cloudyFrag", "dzn_KB", "sel"
    );

    for &n in &sizes {
        if n > n_pool {
            println!("{n:>5}  SKIP (pool has only {n_pool} images)");
            continue;
        }
        let seed = seed_for(name, n, base_seed);
        let sel = sample_instance(&aoi, &pool.mps, &pool.cloud, n, params, seed);
        let idx = sel.image_indices(n);

        let sel_polys: Vec<Polygon<f64>> = idx.iter().map(|&i| pool.polys[i].clone()).collect();
        let frag = fragmentize_within_aoi(&sel_polys, &aoi);
        let areas_m2: Vec<f64> = frag
            .fragments
            .iter()
            .map(|p| p.unsigned_area() * k_area)
            .collect();
        let max_cloud_area = areas_m2.iter().sum();

        let problem = SimsProblem {
            num_images: idx.len(),
            universe: frag.fragments.len(),
            images: frag.images_to_fragments,
            costs: idx.iter().map(|&i| pool.cost[i]).collect(),
            cloud_coverages: idx.iter().map(|&i| pool.cloud[i]).collect(),
            areas: areas_m2,
            resolution: idx.iter().map(|&i| pool.resolution[i]).collect(),
            incidence_angle: idx.iter().map(|&i| pool.incidence[i]).collect(),
            max_cloud_area,
        };

        let disc = problem.discretize(seed ^ 0xC10D_C10D_C10D_C10D);
        if let Err(e) = disc.validate() {
            println!("{n:>5}  INVALID: {e}");
            continue;
        }

        let dzn_path = Path::new(out_dir).join(format!("{name}_{n}.dzn"));
        disc.to_dzn(&dzn_path).unwrap();
        let cloudy: usize = disc.clouds.iter().map(Vec::len).sum();
        let kb = std::fs::metadata(&dzn_path)
            .map(|m| m.len() as f64 / 1024.0)
            .unwrap_or(0.0);

        // Sidecar metadata (image ids + selection diagnostics).
        let cov_pct = 100.0 * sel.covered_cells as f64 / sel.num_cells.max(1) as f64;
        let sel_cloud_mean =
            idx.iter().map(|&i| pool.cloud[i]).sum::<f64>() / idx.len().max(1) as f64;
        let meta = serde_json::json!({
            "name": name, "N": n, "seed": seed, "universe": disc.universe,
            "backbone_size": sel.backbone_size, "coverage_pct": cov_pct,
            "grid_cells": sel.num_cells, "cloudy_assignments": cloudy,
            "sel_cloud_mean": sel_cloud_mean,
            "cost_sum": disc.costs.iter().sum::<i64>(),
        });
        std::fs::write(
            Path::new(out_dir).join(format!("{name}_{n}.meta.json")),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();

        println!(
            "{n:>5} {:>9} {:>9} {cloudy:>10} {kb:>8.1} {:>8}",
            disc.universe,
            sel.backbone_size,
            idx.len()
        );
    }
    println!("wrote instances to {out_dir}/");
}
