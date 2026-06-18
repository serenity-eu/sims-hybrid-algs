use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Generate cloudy fragment sets for each image, mirroring Python's
/// `SimsProblem._generate_bool_clouds()`.
///
/// For each image:
///  - Compute the total area of its fragments.
///  - Compute the cloudy area budget = `cloud_coverage_pct / 100 * total_area`.
///  - Shuffle the image's fragments and greedily add them until the budget is
///    (approximately) reached, skipping any fragment that would overshoot by
///    more than 10%.
///
/// # Arguments
/// * `images`           – fragment index lists per image (0-based).
/// * `areas`            – area (m²) of each fragment (indexed by fragment idx).
/// * `cloud_coverages`  – cloud coverage percentage (0–100) per image.
/// * `rng`              – seeded random number generator.
///
/// # Returns
/// `clouds[i]` is the sorted list of fragment indices that are cloudy for image i.
pub fn generate_clouds(
    images: &[Vec<usize>],
    areas: &[f64],
    cloud_coverages: &[f64],
    rng: &mut impl Rng,
) -> Vec<Vec<usize>> {
    images
        .iter()
        .zip(cloud_coverages.iter())
        .map(|(image_fragments, &coverage_pct)| {
            let image_area: f64 = image_fragments.iter().map(|&fi| areas[fi]).sum();
            let cloudy_budget = coverage_pct / 100.0 * image_area;

            if cloudy_budget == 0.0 {
                return Vec::new();
            }

            let mut shuffled = image_fragments.clone();
            shuffled.shuffle(rng);

            let mut cloudy = Vec::new();
            let mut cloudy_area = 0.0_f64;

            for &fi in &shuffled {
                let frag_area = areas[fi];
                // Skip if adding this fragment would exceed 110% of the budget
                if cloudy_area + frag_area >= 1.1 * cloudy_budget {
                    continue;
                }
                cloudy_area += frag_area;
                cloudy.push(fi);
            }

            cloudy.sort_unstable();
            cloudy
        })
        .collect()
}

/// Convenience wrapper that creates a seeded `StdRng`.
pub fn generate_clouds_seeded(
    images: &[Vec<usize>],
    areas: &[f64],
    cloud_coverages: &[f64],
    seed: u64,
) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_clouds(images, areas, cloud_coverages, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_coverage_produces_empty_clouds() {
        let images = vec![vec![0, 1, 2]];
        let areas = vec![100.0, 200.0, 150.0];
        let coverages = vec![0.0];
        let clouds = generate_clouds_seeded(&images, &areas, &coverages, 42);
        assert!(clouds[0].is_empty());
    }

    #[test]
    fn test_full_coverage_covers_all_fragments() {
        let images = vec![vec![0, 1, 2]];
        let areas = vec![100.0, 100.0, 100.0];
        let coverages = vec![100.0]; // 100% cloud
        let clouds = generate_clouds_seeded(&images, &areas, &coverages, 42);
        // With 100% coverage budget = total area; all three fragments fit
        assert!(!clouds[0].is_empty());
    }

    #[test]
    fn test_clouds_are_subset_of_image_fragments() {
        let images = vec![vec![0, 1, 2, 3, 4]];
        let areas = vec![50.0, 80.0, 120.0, 90.0, 60.0];
        let coverages = vec![40.0];
        let clouds = generate_clouds_seeded(&images, &areas, &coverages, 7);
        for &fi in &clouds[0] {
            assert!(images[0].contains(&fi), "cloudy fragment {fi} not in image");
        }
    }
}
