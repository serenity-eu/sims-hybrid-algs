use rand::{distributions::Open01, Rng};

pub mod objectives_utils {
    use super::*;
    
    pub fn generate_weights<const D: usize>() -> [f32; D] {
        let mut weights = [0.0f32; D];
        let mut remaining = 1.0_f32;
        
        for i in 0..D-1 {
            let weight: f32 = rand::thread_rng().sample(Open01);
            weights[i] = weight * remaining;
            remaining -= weight * remaining;
        }
        weights[D-1] = remaining; // Last weight gets the remainder
        weights
    }

    // Legacy function for 2D backwards compatibility
    pub fn generate_weights_2d() -> (f32, f32) {
        let weights = generate_weights::<2>();
        (weights[0], weights[1])
    }

    pub fn weighted_sum<const D: usize>(objectives: &pareto::Objectives<D>, weights: &[f32; D]) -> f32 {
        objectives.iter().zip(weights.iter())
            .map(|(&obj, &weight)| obj as f32 * weight)
            .sum()
    }

    // Legacy function for 2D backwards compatibility
    pub fn weighted_sum_2d<const D: usize>(objectives: &pareto::Objectives<D>, weights: (f32, f32), _max_values: pareto::Objectives<D>) -> f32 {
        assert_eq!(D, 2, "weighted_sum_2d only supports 2D objectives");
        objectives[0] as f32 * weights.0 + objectives[1] as f32 * weights.1
    }

    pub fn apply_delta<const D: usize>(objectives: &mut pareto::Objectives<D>, deltas: &[i64; D]) {
        for (i, &delta) in deltas.iter().enumerate() {
            if delta < 0 {
                objectives[i] -= delta.unsigned_abs();
            } else {
                objectives[i] += delta as u64;
            }
        }
    }

    // Legacy function for 2D backwards compatibility
    pub fn apply_delta_2d<const D: usize>(objectives: &mut pareto::Objectives<D>, delta: (i64, i64)) {
        assert_eq!(D, 2, "apply_delta_2d only supports 2D objectives");
        if delta.0 < 0 {
            objectives[0] -= delta.0.unsigned_abs();
        } else {
            objectives[0] += delta.0 as u64;
        }
        if delta.1 < 0 {
            objectives[1] -= delta.1.unsigned_abs();
        } else {
            objectives[1] += delta.1 as u64;
        }
    }

    pub fn to_tuple<const D: usize>(objectives: &pareto::Objectives<D>) -> (i32, i32) {
        assert_eq!(D, 2, "to_tuple only supports 2D objectives");
        (objectives[0] as i32, objectives[1] as i32)
    }

    pub fn new<const D: usize>(cost: u64, cloudy_area: u64) -> pareto::Objectives<D> {
        assert_eq!(D, 2, "new only supports 2D objectives");
        let mut result = [0u64; D];
        result[0] = cost;
        result[1] = cloudy_area;
        result
    }
}
