"""Quick reproducer for the 4D PLS heap corruption crash on mexico_city_100."""
import sims_problem
from datetime import timedelta
import sys

instance_path = "tests/data/mexico_city_100.dzn"
problem = sims_problem.SimsDiscreteProblem.from_dzn(instance_path)

print(f"Problem: {problem.num_images} images")

# Run multiple times to try to trigger the non-deterministic crash
for run in range(10):
    print(f"\n=== Run {run + 1}/10, deterministic={run % 2 == 0} ===")
    print("Running 4D PLS with nd-tree, 30s timeout...")
    sys.stdout.flush()
    
    try:
        result = sims_problem.solve_with_pls(
            sims_instance=problem,
            objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle", "min_resolution"],
            timeout=timedelta(seconds=30),
            max_iterations=50000,
            is_deterministic=(run % 2 == 0),
            initial_population_size=100,
            trace=False,
            include_dominated=True,
            pareto_archive="nd-tree",
            profiling_trace=False,
        )
        print(f"  Completed with {len(result.final_solutions)} solutions")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        sys.exit(1)

print("\nAll runs completed without crash.")
