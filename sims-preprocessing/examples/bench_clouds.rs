//! Quick throughput check for the random-cloud generator.
//! Run: `cargo run --release --example bench_clouds`
use sims_preprocessing::clouds::generate_clouds;
use std::time::Instant;

fn main() {
    // Realistic shape: N images over a ~5k-fragment universe, ~120 frags/image.
    for &n_images in &[100usize, 300, 500, 2000] {
        let universe = 5000usize;
        let frags_per_image = 120usize;
        let images: Vec<Vec<usize>> = (0..n_images)
            .map(|k| {
                let start = (k * 37) % (universe - frags_per_image);
                (start..start + frags_per_image).collect()
            })
            .collect();
        let areas: Vec<f64> = (0..universe)
            .map(|i| 100.0 + (i % 13) as f64 * 10.0)
            .collect();
        // Cloud coverage 0..100, mirroring the relaxed (no cap) distribution.
        let cov: Vec<f64> = (0..n_images).map(|k| (k as f64 * 7.0) % 100.0).collect();

        // warm up
        let _ = generate_clouds(&images, &areas, &cov, 1);
        let reps = 200;
        let t = Instant::now();
        let mut checksum = 0usize;
        for s in 0..reps {
            let clouds = generate_clouds(&images, &areas, &cov, s);
            checksum ^= clouds.iter().map(Vec::len).sum::<usize>();
        }
        let per = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        println!(
            "n_images={n_images:>5}  universe={universe}  ~{frags_per_image} frags/img  \
             -> {per:8.1} µs/instance  (checksum {checksum})"
        );
    }
}
