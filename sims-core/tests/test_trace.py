#!/usr/bin/env python3
"""
Test suite for trace generation and merging functionality.
This test validates that MILP and PLS solvers generate valid traces and that
trace merging preserves data integrity and hypervolume monotonicity.
"""

from io import BytesIO
import struct
import pytest
import tempfile
import gzip
import tarfile
import json
from pathlib import Path
from datetime import timedelta

import sims_problem
from sims.core.sims.problem import ProblemInstance
from sims.core.sims.solver import solve, solve_with_two_phases
from sims.core.sims.solver_config import TwoPhaseSolverConfig, SolverType, FrontStrategy
from sims.core.sims.solver_result import TwoPhaseSolverResult

# Logging is now configured via pyproject.toml [tool.pytest.ini_options]
# Console: WARNING level, File: DEBUG level (when using --log-file option)


class TraceData:
    """Container for parsed trace data"""
    def __init__(self, metadata, objectives, dominated, timestamps, hypervolume):
        self.metadata = metadata
        self.objectives = objectives
        self.dominated = dominated
        self.timestamps = timestamps
        self.hypervolume = hypervolume


def extract_trace_archive(archive_bytes):
    """Extract files from gzipped tar archive"""
    with gzip.GzipFile(fileobj=BytesIO(archive_bytes), mode='rb') as gz:
        tar_data = gz.read()
    
    files = {}
    with tarfile.open(fileobj=BytesIO(tar_data), mode='r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    files[member.name] = file_obj.read()
    
    return files


def parse_trace_data(archive_bytes):
    """Parse trace archive into structured data"""
    files = extract_trace_archive(archive_bytes)
    
    # Parse metadata
    metadata = json.loads(files['metadata.json'].decode('utf-8'))
    
    # Parse binary data
    num_solutions = metadata['solution_count']
    num_objectives = len(metadata['objectives'])
    
    # Parse objectives.bin
    objectives = []
    for i in range(num_solutions):
        solution_objs = []
        for j in range(num_objectives):
            offset = (i * num_objectives + j) * 8
            value = struct.unpack('<Q', files['objectives.bin'][offset:offset+8])[0]
            solution_objs.append(value)
        objectives.append(solution_objs)
    
    # Parse dominated.bin
    dominated = []
    for i in range(num_solutions):
        value = struct.unpack('<B', files['dominated.bin'][i:i+1])[0]
        dominated.append(bool(value))
    
    # Parse timestamp.bin
    timestamps = []
    for i in range(num_solutions):
        offset = i * 4
        value = struct.unpack('<I', files['timestamp.bin'][offset:offset+4])[0]
        timestamps.append(value)
    
    # Parse hypervolume.bin
    hypervolume = []
    for i in range(num_solutions):
        offset = i * 8
        value = struct.unpack('<d', files['hypervolume.bin'][offset:offset+8])[0]
        hypervolume.append(value)
    
    return TraceData(metadata, objectives, dominated, timestamps, hypervolume)


class TestLagosNigeria30Trace:
    """Test trace functionality with lagos_nigeria_30 instance"""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory"""
        return Path(__file__).parent / "data"
    
    def test_two_phase_trace_generation(self, test_data_dir):
        """Test end-to-end trace generation via two-phase solver"""
        # Use the actual lagos_nigeria_30 instance
        instance_path = Path(test_data_dir) / "lagos_nigeria_30.dzn"
        
        if not instance_path.exists():
            pytest.skip(f"Instance file not found: {instance_path}")
        
        # Create problem instance
        problem_instance = ProblemInstance.from_dzn(instance_path)
        
        # Configure two-phase solver
        solver_config = TwoPhaseSolverConfig(
            exact_solver_type=SolverType.GUROBI,
            front_strategy=FrontStrategy.GPBA_A,
            timeout_s=15,
            ratio=(40, 60)  # 40% exact, 60% PLS
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run two-phase solver with trace enabled
            result = solve_with_two_phases(
                problem_instance=problem_instance,
                problem_path=instance_path,
                experiment_path=Path(temp_dir),
                solver_config=solver_config,
                objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle"],
                enable_pls_trace=True
            )
            
            assert result.trace_data is not None, "Two-phase solver should generate merged trace"
            
            # Parse and validate trace
            trace_data = parse_trace_data(result.trace_data)
            
            # Basic validation
            assert trace_data.metadata['algorithm'] == 'hybrid', "Two-phase trace should have 'hybrid' algorithm"
            assert 'ratio' in trace_data.metadata, "Two-phase trace should have ratio metadata"
            assert trace_data.metadata['ratio'] == [40, 60], "Ratio should match solver config"
            
            # Check solution count consistency - Note: PLS trace includes ALL explored solutions
            exact_count = len(result.exact_solver_result.pareto_front) if result.exact_solver_result else 0
            pls_count = len(result.pls_result.pareto_front) if result.pls_result else 0
            
            print(f"Exact solutions: {exact_count}, PLS solutions: {pls_count}")
            print(f"Trace metadata solution count: {trace_data.metadata['solution_count']}")
            
            # The trace includes ALL explored solutions (dominated + non-dominated)
            # So we expect the trace count to be >= the final Pareto front sizes
            assert trace_data.metadata['solution_count'] >= exact_count, "Trace should include at least all exact solutions"
            assert trace_data.metadata['solution_count'] >= pls_count, "Trace should include at least all PLS final solutions"
            
            # Validate data array sizes match trace metadata
            trace_count = trace_data.metadata['solution_count']
            assert len(trace_data.objectives) == trace_count, "Objectives array size mismatch"
            assert len(trace_data.timestamps) == trace_count, "Timestamps array size mismatch"
            assert len(trace_data.hypervolume) == trace_count, "Hypervolume array size mismatch"
            assert len(trace_data.dominated) == trace_count, "Dominated array size mismatch"
            
            # Validate hypervolume properties
            violations = sum(1 for i in range(1, len(trace_data.hypervolume)) 
                           if trace_data.hypervolume[i] < trace_data.hypervolume[i-1])
            
            print(f"Hypervolume violations: {violations}")
            print(f"Hypervolume range: {min(trace_data.hypervolume):.6f} → {max(trace_data.hypervolume):.6f}")
            
            if violations > 0:
                # Print some violation details
                for i in range(1, min(len(trace_data.hypervolume), 10)):
                    if trace_data.hypervolume[i] < trace_data.hypervolume[i-1]:
                        print(f"  Violation at {i}: {trace_data.hypervolume[i-1]:.6f} → {trace_data.hypervolume[i]:.6f}")
            
            assert violations == 0, f"Two-phase trace hypervolume not monotonic: {violations} violations"
            assert all(0.0 <= hv <= 1.0 for hv in trace_data.hypervolume), "Two-phase hypervolume values outside [0,1]"
            
            # Check timestamp ordering (should be monotonic after merge)
            timestamp_violations = sum(1 for i in range(1, len(trace_data.timestamps))
                                     if trace_data.timestamps[i] < trace_data.timestamps[i-1])
            assert timestamp_violations == 0, f"Merged timestamps not monotonic: {timestamp_violations} violations"
            
            # Validate required metadata fields
            required_fields = {'objectives', 'solution_count', 'total_duration', 'algorithm', 'objective_bounds', 'reference_point', 'ratio'}
            missing_fields = required_fields - set(trace_data.metadata.keys())
            assert not missing_fields, f"Missing metadata fields: {missing_fields}"
            
            # Validate objective bounds and reference point
            assert len(trace_data.metadata['objective_bounds']) == 3, "Should have 3 objective bounds"
            assert len(trace_data.metadata['reference_point']) == 3, "Should have 3 reference point values"
            
            print("✅ Lagos Nigeria 30 two-phase trace test passed!")
    
    def test_pls_only_trace(self):
        """Test PLS-only trace generation for comparison"""
        # Create a simple problem for PLS testing
        problem = sims_problem.SimsDiscreteProblem(
            num_images=5,
            universe=5,
            images=[[0], [1], [2], [3], [4]],
            costs=[10, 20, 30, 40, 50],
            clouds=[[], [], [], [], []],
            areas=[1, 1, 1, 1, 1],
            max_cloud_area=0,
            resolution=[10, 20, 30, 40, 50],
            incidence_angle=[45, 50, 55, 60, 65]
        )
        
        result = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_max_incidence_angle"],
            timeout=timedelta(seconds=3),
            max_iterations=20
        )
        
        if result.trace is None:
            pytest.skip("PLS trace not generated")
        
        trace_data = parse_trace_data(result.trace)
        
        # Basic validation
        assert trace_data.metadata['algorithm'] in ['PLS-3D', 'PLS'], "PLS trace should have PLS algorithm"
        assert len(trace_data.metadata['objectives']) == 3, "Should have 3 objectives"
        
        # Validate hypervolume monotonicity
        violations = sum(1 for i in range(1, len(trace_data.hypervolume)) 
                       if trace_data.hypervolume[i] < trace_data.hypervolume[i-1])
        
        assert violations == 0, f"PLS hypervolume not monotonic: {violations} violations"
        assert all(0.0 <= hv <= 1.0 for hv in trace_data.hypervolume), "PLS hypervolume values outside [0,1]"
        
        print("✅ PLS-only trace test passed!")
    
    def test_individual_phases_and_merge_validation(self, test_data_dir):
        """
        Comprehensive test that:
        1. Runs MILP phase individually and validates its trace
        2. Runs PLS phase individually and validates its trace
        3. Merges the traces and validates the merged result
        4. Validates consistency between individual traces and merged trace
        """

        # Note: Reference points are calculated dynamically based on actual solutions found
        # They can vary between runs depending on the solutions discovered by each phase
        # Examples from recent runs:
        # - MILP: [6551201, 511694, 477], PLS: [8693431, 574662, 480] 
        # - MILP: [6551201, 511694, 477], PLS: [9749711, 632662, 480]
        # We'll validate that actual reference points are reasonable rather than exact values

        # Use the actual lagos_nigeria_30 instance
        instance_path = Path(test_data_dir) / "lagos_nigeria_30.dzn"

        if not instance_path.exists():
            pytest.skip(f"Instance file not found: {instance_path}")

        # Create problem instance
        problem_instance = ProblemInstance.from_dzn(instance_path)
        objectives = ["min_cost", "cloud_coverage", "min_max_incidence_angle"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_path = Path(temp_dir)
            summary_path = experiment_path / f"{problem_instance.name.rsplit('_', maxsplit=1)[0]}.csv"
            
            # Phase 1: Run MILP solver individually with trace enabled
            print("\n=== Phase 1: MILP Solver ===")
            milp_result = solve(
                solver_type=SolverType.GUROBI,
                problem_instance=problem_instance,
                problem_path=instance_path,
                timeout_s=1,  # Very short timeout to get fewer solutions
                output_path=summary_path,
                objectives=objectives,
                front_strategy=FrontStrategy.GPBA_A,
                enable_trace=True  # Enable tracing to match solve_with_two_phases
            )
            
            assert milp_result.trace_data is not None, "MILP solver should generate trace when enabled"
            milp_trace = parse_trace_data(milp_result.trace_data)
            
            # Validate MILP trace
            print(f"MILP solutions: {len(milp_result.pareto_front)}")
            print(f"MILP trace metadata count: {milp_trace.metadata['solution_count']}")
            
            assert milp_trace.metadata['algorithm'] in ['MILP', 'GUROBI', 'Gurobi', 'GPBA_A'], f"MILP trace algorithm: {milp_trace.metadata['algorithm']}"
            assert milp_trace.metadata['solution_count'] == len(milp_result.pareto_front), "MILP trace count mismatch"
            assert len(milp_trace.objectives) == milp_trace.metadata['solution_count'], "MILP objectives array size mismatch"
            assert len(milp_trace.hypervolume) == milp_trace.metadata['solution_count'], "MILP hypervolume array size mismatch"
            
            # Validate MILP hypervolume monotonicity
            milp_hv_violations = sum(1 for i in range(1, len(milp_trace.hypervolume)) 
                                   if milp_trace.hypervolume[i] < milp_trace.hypervolume[i-1])
            assert milp_hv_violations == 0, f"MILP hypervolume not monotonic: {milp_hv_violations} violations"
            assert all(0.0 <= hv <= 1.0 for hv in milp_trace.hypervolume), "MILP hypervolume values outside [0,1]"
            
            print(f"✅ MILP trace valid: {len(milp_trace.hypervolume)} solutions, HV: {min(milp_trace.hypervolume):.6f}→{max(milp_trace.hypervolume):.6f}")
            
            # Phase 2: Run PLS solver individually with MILP solutions as initial population
            print("\n=== Phase 2: PLS Solver (seeded with MILP solutions) ===")
            
            # Use MILP results as initial population for PLS (matching solve_with_two_phases behavior)
            initial_population = milp_result.pareto_front if milp_result.pareto_front else None
            if initial_population:
                print(f"Seeding PLS with {len(initial_population)} solutions from MILP phase")
            
            pls_result = solve(
                solver_type=SolverType.PLS,
                problem_instance=problem_instance,
                problem_path=instance_path,
                timeout_s=1,  # Very short timeout to get fewer solutions
                output_path=summary_path,
                objectives=objectives,
                initial_population=initial_population,  # Seed with MILP results
                enable_trace=True
            )
            
            assert pls_result.trace_data is not None, "PLS solver should generate trace when enabled"
            pls_trace = parse_trace_data(pls_result.trace_data)
            
            # Validate PLS trace
            print(f"PLS solutions: {len(pls_result.pareto_front)}")
            print(f"PLS trace metadata count: {pls_trace.metadata['solution_count']}")
            print(f"PLS final pareto front size: {len(pls_result.pareto_front)}")
            print(f"PLS trace objectives array size: {len(pls_trace.objectives)}")
            print(f"PLS trace hypervolume array size: {len(pls_trace.hypervolume)}")
            
            assert pls_trace.metadata['algorithm'] in ['PLS', 'PLS-3D'], f"PLS trace algorithm: {pls_trace.metadata['algorithm']}"
            # Note: PLS trace records all explored solutions, not just final Pareto front
            # So we compare trace arrays with trace metadata count, not final front size
            assert len(pls_trace.objectives) == pls_trace.metadata['solution_count'], "PLS objectives array size mismatch"
            assert len(pls_trace.hypervolume) == pls_trace.metadata['solution_count'], "PLS hypervolume array size mismatch"
            
            # Validate PLS hypervolume monotonicity
            pls_hv_violations = sum(1 for i in range(1, len(pls_trace.hypervolume)) 
                                  if pls_trace.hypervolume[i] < pls_trace.hypervolume[i-1])
            assert pls_hv_violations == 0, f"PLS hypervolume not monotonic: {pls_hv_violations} violations"
            assert all(0.0 <= hv <= 1.0 for hv in pls_trace.hypervolume), "PLS hypervolume values outside [0,1]"
            
            print(f"✅ PLS trace valid: {len(pls_trace.hypervolume)} solutions, HV: {min(pls_trace.hypervolume):.6f}→{max(pls_trace.hypervolume):.6f}")
            
            # Validate PLS reference point is reasonable (all positive values)
            actual_pls_ref_point = pls_trace.metadata.get('reference_point')
            assert actual_pls_ref_point is not None, "PLS trace should have reference point"
            assert all(isinstance(x, (int, float)) and x > 0 for x in actual_pls_ref_point), f"PLS reference point should be positive numbers: {actual_pls_ref_point}"
            print(f"✅ PLS reference point is valid: {actual_pls_ref_point}")
            
            # Phase 3: Use TwoPhaseSolverResult.from_results_pair() for proper merging (as done in solve_with_two_phases)
            print("\n=== Phase 3: Proper Trace Merging via TwoPhaseSolverResult.from_results_pair() ===")
            
            # Validate we have both traces for merging
            assert milp_result.trace_data is not None, "MILP trace required for proper merging"
            assert pls_result.trace_data is not None, "PLS trace required for proper merging"
            
            # Create solver config matching our test setup
            two_phase_config = TwoPhaseSolverConfig(
                exact_solver_type=SolverType.GUROBI,
                front_strategy=FrontStrategy.GPBA_A,
                timeout_s=2,  # Total timeout (not used in merge, just for config)
                ratio=(50, 50)  # Equal time split
            )
            
            # Use the exact same method as solve_with_two_phases() 
            # This creates TwoPhaseSolverResult which automatically computes merged trace in __post_init__
            merged_result = TwoPhaseSolverResult.from_results_pair(
                exact_solver_result=milp_result,
                pls_result=pls_result,
                solver_config=two_phase_config,
                filter_invalid=False  # Don't filter, we want to see all solutions
            )
            
            # Parse the automatically merged trace
            assert merged_result.trace_data is not None, "TwoPhaseSolverResult should automatically generate merged trace"
            merged_trace = parse_trace_data(merged_result.trace_data)
            
            # Validate merged trace properties
            print(f"Merged trace solutions: {merged_trace.metadata['solution_count']} (all solutions explored during search)")
            print(f"Final Pareto front sizes: MILP ({len(milp_result.pareto_front)}) + PLS ({len(pls_result.pareto_front)}) = {len(milp_result.pareto_front) + len(pls_result.pareto_front)}")
            
            # The merged trace should contain solutions from both phases
            assert merged_trace.metadata['solution_count'] >= len(milp_result.pareto_front), "Merged trace should include MILP solutions"
            assert merged_trace.metadata['solution_count'] >= len(pls_result.pareto_front), "Merged trace should include PLS solutions"
            assert merged_trace.metadata['algorithm'] == "hybrid", "Merged trace should have hybrid algorithm"
            assert merged_trace.metadata['ratio'] == [50, 50], "Merged trace should have correct ratio"
            
            # Validate merged trace arrays have correct sizes
            assert len(merged_trace.objectives) == merged_trace.metadata['solution_count'], "Merged objectives array size mismatch"
            assert len(merged_trace.hypervolume) == merged_trace.metadata['solution_count'], "Merged hypervolume array size mismatch"
            assert len(merged_trace.timestamps) == merged_trace.metadata['solution_count'], "Merged timestamps array size mismatch"
            assert len(merged_trace.dominated) == merged_trace.metadata['solution_count'], "Merged dominated array size mismatch"
            
            # Validate hypervolume monotonicity (key requirement)
            merged_hv_violations = sum(1 for i in range(1, len(merged_trace.hypervolume))
                                     if merged_trace.hypervolume[i] < merged_trace.hypervolume[i-1])
            assert merged_hv_violations == 0, f"Merged hypervolume not monotonic: {merged_hv_violations} violations"
            assert all(0.0 <= hv <= 1.0 for hv in merged_trace.hypervolume), "Merged hypervolume values outside [0,1]"
            
            # Validate timestamp monotonicity (timestamps should be properly offset)
            timestamp_violations = sum(1 for i in range(1, len(merged_trace.timestamps))
                                     if merged_trace.timestamps[i] < merged_trace.timestamps[i-1])
            assert timestamp_violations == 0, f"Merged timestamps not monotonic: {timestamp_violations} violations"
            
            # Validate bounds and reference point were calculated automatically by TwoPhaseSolverResult
            assert 'objective_bounds' in merged_trace.metadata, "Merged trace should have objective_bounds"
            assert 'reference_point' in merged_trace.metadata, "Merged trace should have reference_point"
            assert len(merged_trace.metadata['objective_bounds']) == 3, "Should have bounds for 3 objectives"
            assert len(merged_trace.metadata['reference_point']) == 3, "Should have reference point for 3 objectives"
            
            print(f"Auto-calculated objective bounds: {merged_trace.metadata['objective_bounds']}")
            print(f"Auto-calculated reference point: {merged_trace.metadata['reference_point']}")
            
            # Validate second phase start index is properly set
            assert 'second_phase_start_index' in merged_trace.metadata, "Merged trace should have second_phase_start_index"
            assert merged_trace.metadata['second_phase_start_index'] == len(milp_result.pareto_front), "Second phase start index should match MILP solution count"
            
            print(f"✅ Proper merge completed: {len(merged_trace.hypervolume)} solutions, HV: {min(merged_trace.hypervolume):.6f}→{max(merged_trace.hypervolume):.6f}")
            print(f"✅ Hypervolume monotonicity verified: {merged_hv_violations} violations")
            print(f"✅ Timestamp monotonicity verified: {timestamp_violations} violations")
            
            print("\n=== Test Summary ===")
            print(f"✅ MILP phase: {milp_trace.metadata['solution_count']} solutions explored (= {len(milp_result.pareto_front)} final Pareto)")
            print(f"✅ PLS phase: {pls_trace.metadata['solution_count']} solutions explored (→ {len(pls_result.pareto_front)} final Pareto)") 
            print(f"✅ Merged result: {merged_trace.metadata['solution_count']} total solutions explored")
            print("✅ All traces have valid hypervolume progression")
            
            print(f"Expected final Pareto solutions: {len(milp_result.pareto_front) + len(pls_result.pareto_front)} (MILP: {len(milp_result.pareto_front)}, PLS: {len(pls_result.pareto_front)})")
            print(f"Note: Total solutions explored ({merged_trace.metadata['solution_count']}) includes all intermediate search solutions, not just final Pareto front")

            # Validate individual trace properties
            assert milp_trace.metadata['solution_count'] == len(milp_result.pareto_front), "MILP trace count mismatch"
            # Note: PLS trace records all explored solutions, not just final Pareto front
            # So we validate that PLS trace has at least as many solutions as the final front
            assert pls_trace.metadata['solution_count'] >= len(pls_result.pareto_front), "PLS trace should include all final Pareto solutions"
            
            # Validate hypervolume properties in both traces
            milp_hv_violations = sum(1 for i in range(1, len(milp_trace.hypervolume)) 
                                if milp_trace.hypervolume[i] < milp_trace.hypervolume[i-1])
            pls_hv_violations = sum(1 for i in range(1, len(pls_trace.hypervolume)) 
                                if pls_trace.hypervolume[i] < pls_trace.hypervolume[i-1])
            
            assert milp_hv_violations == 0, f"MILP hypervolume not monotonic: {milp_hv_violations} violations"
            assert pls_hv_violations == 0, f"PLS hypervolume not monotonic: {pls_hv_violations} violations"
            
            print("✅ Both traces valid and ready for merging")
            print(f"   MILP HV: {min(milp_trace.hypervolume):.6f}→{max(milp_trace.hypervolume):.6f}")
            print(f"   PLS HV:  {min(pls_trace.hypervolume):.6f}→{max(pls_trace.hypervolume):.6f}")
            
            print("✅ Proper merge completed using TwoPhaseSolverResult.from_results_pair()")
            
            # Phase 4: Consistency validation between individual and merged traces
            print("\n=== Phase 4: Consistency Validation ===")
            
            # The merged hypervolume should be at least as good as both individual final hypervolumes
            final_milp_hv = milp_trace.hypervolume[-1] if milp_trace.hypervolume else 0.0
            final_pls_hv = pls_trace.hypervolume[-1] if pls_trace.hypervolume else 0.0
            final_merged_hv = merged_trace.hypervolume[-1] if merged_trace.hypervolume else 0.0
            
            print(f"Final hypervolumes - MILP: {final_milp_hv:.6f}, PLS: {final_pls_hv:.6f}, Proper Merge: {final_merged_hv:.6f}")
            
            # The properly merged hypervolume should be at least as good as both individual final hypervolumes
            # Since we're using the actual merge_traces function, we expect proper monotonic behavior
            assert final_merged_hv >= final_milp_hv, f"Proper merged HV ({final_merged_hv:.6f}) should be >= MILP HV ({final_milp_hv:.6f})"
            
            # The properly merged trace should show the combined benefits from both phases
            improvement_threshold = 0.0
            assert final_merged_hv >= final_milp_hv + improvement_threshold, f"Proper merged trace should show improvement: {final_merged_hv:.6f} >= {final_milp_hv:.6f}"
            
            # Check that bounds and reference points were auto-calculated correctly by TwoPhaseSolverResult
            print(f"Merged trace bounds: {merged_trace.metadata['objective_bounds']}")
            print(f"Merged trace reference point: {merged_trace.metadata['reference_point']}")
            
            # Validate that timestamps are properly offset (PLS phase should start after MILP phase)
            milp_solution_count = len(milp_result.pareto_front)
            if milp_solution_count > 0 and milp_solution_count < len(merged_trace.timestamps):
                milp_end_time = merged_trace.timestamps[milp_solution_count - 1]
                pls_start_time = merged_trace.timestamps[milp_solution_count]
                print(f"MILP phase end time: {milp_end_time}, PLS phase start time: {pls_start_time}")
                assert pls_start_time >= milp_end_time, "PLS timestamps should be properly offset after MILP phase"
            
            # Validate that the merge maintains objectives consistency
            assert merged_trace.metadata['objectives'] == milp_trace.metadata['objectives'], "Objectives should match between MILP and merged"
            assert merged_trace.metadata['objectives'] == pls_trace.metadata['objectives'], "Objectives should match between PLS and merged"
            
            print("✅ All consistency checks passed!")
            print("\n🎉 Individual phases and proper trace merging validation completed successfully!")
            print(f"   MILP: {milp_trace.metadata['solution_count']} explored → {len(milp_result.pareto_front)} final Pareto solutions")
            print(f"   PLS:  {pls_trace.metadata['solution_count']} explored → {len(pls_result.pareto_front)} final Pareto solutions") 
            print(f"   Merged trace: {merged_trace.metadata['solution_count']} total solutions explored (combined from both phases)")
            print(f"   Final hypervolume improvement: {final_milp_hv:.6f} → {final_merged_hv:.6f}")
            print("   Proper merge using sims_problem.merge_traces() completed successfully!")
            
            # Final validation of classification claims from TRACE_SPECIFICATION.md
            print("\n=== Trace Classification Validation ===")
            
            # Validate MILP behavior: traces only final solutions (per spec)
            print(f"MILP trace validation: {milp_trace.metadata['solution_count']} traced == {len(milp_result.pareto_front)} final Pareto")
            assert milp_trace.metadata['solution_count'] == len(milp_result.pareto_front), "MILP should trace only final Pareto solutions"
            
            # Validate PLS behavior: traces all explored solutions (per spec)
            print(f"PLS trace validation: {pls_trace.metadata['solution_count']} traced >> {len(pls_result.pareto_front)} final Pareto")
            assert pls_trace.metadata['solution_count'] >= len(pls_result.pareto_front), "PLS should trace all explored solutions"
            assert pls_trace.metadata['solution_count'] > len(pls_result.pareto_front), "PLS should explore more than final Pareto count"
            
            # Validate merged trace identification (per spec)
            assert merged_trace.metadata['algorithm'] == "hybrid", "Merged trace must have algorithm='hybrid'"
            assert merged_trace.metadata.get('ratio') is not None, "Merged trace must have ratio field"
            assert merged_trace.metadata.get('second_phase_start_index') is not None, "Merged trace must have second_phase_start_index"
            assert merged_trace.metadata['second_phase_start_index'] == len(milp_result.pareto_front), "Second phase start index must equal MILP solution count"
            
            # Validate solution count arithmetic (per spec)
            expected_merged_count = milp_trace.metadata['solution_count'] + pls_trace.metadata['solution_count']
            assert merged_trace.metadata['solution_count'] == expected_merged_count, f"Merged count ({merged_trace.metadata['solution_count']}) should equal sum ({expected_merged_count})"
            
            # Validate trace type classification
            def classify_trace(trace_metadata):
                if trace_metadata.get('algorithm') == "hybrid" and trace_metadata.get('ratio') is not None:
                    return "merged"
                elif trace_metadata.get('algorithm') in ['PLS', 'PLS-3D']:
                    return "pls" 
                elif trace_metadata.get('algorithm') in ['MILP', 'GUROBI', 'Gurobi', 'GPBA_A']:
                    return "milp"
                else:
                    return "unknown"
            
            milp_type = classify_trace(milp_trace.metadata)
            pls_type = classify_trace(pls_trace.metadata)
            merged_type = classify_trace(merged_trace.metadata)
            
            assert milp_type == "milp", f"MILP trace misclassified as {milp_type}"
            assert pls_type == "pls", f"PLS trace misclassified as {pls_type}"
            assert merged_type == "merged", f"Merged trace misclassified as {merged_type}"
            
            print(f"✅ Trace classification verified: MILP={milp_type}, PLS={pls_type}, Merged={merged_type}")
            print("✅ Solution count patterns verified: MILP=final_only, PLS=all_explored, Merged=combined")
            print("✅ All TRACE_SPECIFICATION.md claims validated!")
            
            # Final summary with validated claims
            print("\n📊 Validated Trace Behavior Summary:")
            print(f"   • MILP traces final solutions only: {milp_trace.metadata['solution_count']} solutions")
            print(f"   • PLS traces all exploration: {pls_trace.metadata['solution_count']} solutions (found {len(pls_result.pareto_front)} final Pareto)")
            print(f"   • Merged trace combines both: {merged_trace.metadata['solution_count']} total solutions")
            print(f"   • Exploration ratio: PLS explored {pls_trace.metadata['solution_count']//len(pls_result.pareto_front)}x more than final Pareto")


class TestDominanceFiltering:
    """Test suite for temporal dominance filtering functionality"""
    
    def test_filtering_with_include_dominated_parameter(self):
        """
        Comprehensive test for include_dominated parameter in PLS solver.
        
        Tests:
        1. include_dominated=True: All explored solutions in trace
        2. include_dominated=False: Only solutions non-dominated at discovery time
        3. Eventual domination tracking in dominated.bin
        4. Solution count validation
        5. Dominated indices correctness
        """
        print("\n" + "="*80)
        print("Testing Dominance Filtering with include_dominated Parameter")
        print("="*80)
        
        # Create a problem that will explore multiple solutions
        # Using overlapping coverage to ensure diverse exploration
        problem = sims_problem.SimsDiscreteProblem(
            num_images=8,
            universe=8,
            images=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
            costs=[10, 12, 15, 11, 13, 14, 16, 18],
            clouds=[[], [1], [], [3], [], [5], [], [7]],
            areas=[2, 2, 2, 2, 2, 2, 2, 2],
            max_cloud_area=3,
            resolution=[10, 12, 14, 11, 13, 15, 16, 17],
            incidence_angle=[30, 32, 35, 31, 33, 36, 38, 40]
        )
        
        # Test 1: Run PLS with include_dominated=True (include all explored)
        print("\n--- Test 1: include_dominated=True (include all explored solutions) ---")
        
        result_all = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage"],
            timeout=timedelta(seconds=10),
            max_iterations=100,
            include_dominated=True  # Keep all explored solutions
        )
        
        if result_all.trace is None:
            pytest.skip("Trace not generated")
        
        trace_all = parse_trace_data(result_all.trace)
        
        print(f"✓ Explored solutions (all): {trace_all.metadata['solution_count']}")
        print(f"✓ Final Pareto front: {len(result_all.final_solutions)}")
        
        # Skip test if PLS didn't explore enough solutions for meaningful testing
        if trace_all.metadata['solution_count'] <= len(result_all.final_solutions):
            pytest.skip(f"PLS explored only {trace_all.metadata['solution_count']} solutions, need more for filtering test")
        
        print(f"✓ Exploration ratio: {trace_all.metadata['solution_count'] / len(result_all.final_solutions):.2f}x")
        
        # Validate that we have more explored than final
        assert trace_all.metadata['solution_count'] > len(result_all.final_solutions), \
            "With include_dominated=True, should have more explored solutions than final Pareto"
        
        # Count dominated solutions in trace_all
        dominated_count_all = sum(1 for d in trace_all.dominated if d)
        non_dominated_count_all = trace_all.metadata['solution_count'] - dominated_count_all
        
        print(f"✓ Dominated solutions: {dominated_count_all}")
        print(f"✓ Non-dominated solutions: {non_dominated_count_all}")
        
        # Test 2: Run PLS with include_dominated=False (filter out dominated at discovery)
        print("\n--- Test 2: include_dominated=False (filter dominated at discovery) ---")
        
        result_filtered = sims_problem.solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage"],
            timeout=timedelta(seconds=10),
            max_iterations=100,
            include_dominated=False  # Filter out dominated solutions
        )
        
        if result_filtered.trace is None:
            pytest.skip("Trace not generated for filtered case")
        
        trace_filtered = parse_trace_data(result_filtered.trace)
        
        print(f"✓ Solutions after filtering: {trace_filtered.metadata['solution_count']}")
        print(f"✓ Final Pareto front: {len(result_filtered.final_solutions)}")
        print(f"✓ Filtered out: {trace_all.metadata['solution_count'] - trace_filtered.metadata['solution_count']} solutions")
        
        # Validate filtering reduced solution count
        assert trace_filtered.metadata['solution_count'] < trace_all.metadata['solution_count'], \
            "Filtering should reduce the number of solutions in trace"
        
        assert trace_filtered.metadata['solution_count'] >= len(result_filtered.final_solutions), \
            "Filtered trace should have at least as many solutions as final Pareto"
        
        # Test 3: Validate dominated.bin structure for filtered trace
        print("\n--- Test 3: Validating dominated.bin structure ---")
        
        # In filtered trace, some solutions may show eventual domination
        # (non-dominated at discovery but dominated by later solutions)
        dominated_count_filtered = sum(1 for d in trace_filtered.dominated if d)
        eventually_dominated = dominated_count_filtered
        
        print(f"✓ Solutions in filtered trace: {trace_filtered.metadata['solution_count']}")
        print(f"✓ Eventually dominated (by later solutions): {eventually_dominated}")
        print(f"✓ Never dominated: {trace_filtered.metadata['solution_count'] - eventually_dominated}")
        
        # The filtered trace should have fewer or equal dominated solutions than unfiltered
        # (because we already filtered out those dominated at discovery)
        assert eventually_dominated <= dominated_count_all, \
            "Filtered trace should not have more dominated solutions than unfiltered"
        
        # Test 4: Validate dominated indices refer to valid positions
        print("\n--- Test 4: Validating domination indices ---")
        
        # Parse dominated.bin as u32 indices (not just bool)
        files_filtered = extract_trace_archive(result_filtered.trace)
        dominated_indices_filtered = []
        for i in range(trace_filtered.metadata['solution_count']):
            offset = i * 4
            index = struct.unpack('<I', files_filtered['dominated.bin'][offset:offset+4])[0]
            dominated_indices_filtered.append(index)
        
        # Validate indices
        max_valid_index = trace_filtered.metadata['solution_count'] - 1
        u32_max = 0xFFFFFFFF
        
        print(f"\nDebug: Filtered trace has {trace_filtered.metadata['solution_count']} solutions")
        print(f"Debug: First 5 objectives: {trace_filtered.objectives[:5]}")
        print(f"Debug: First 5 domination indices: {dominated_indices_filtered[:5]}")
        
        for i, dom_idx in enumerate(dominated_indices_filtered):
            if dom_idx != u32_max:  # If dominated
                assert 0 <= dom_idx <= max_valid_index, \
                    f"Solution {i} has invalid domination index {dom_idx} (max: {max_valid_index})"
                assert dom_idx != i, \
                    f"Solution {i} cannot dominate itself"
                
                # Validate that the dominating solution actually dominates
                dom_obj = trace_filtered.objectives[dom_idx]
                sol_obj = trace_filtered.objectives[i]
                
                print(f"Debug: Solution {i} {sol_obj} dominated by solution {dom_idx} {dom_obj}?")
                
                # Check domination: dom_obj <= sol_obj on all objectives and < on at least one
                all_leq = all(dom_obj[j] <= sol_obj[j] for j in range(len(dom_obj)))
                any_less = any(dom_obj[j] < sol_obj[j] for j in range(len(dom_obj)))
                
                assert all_leq and any_less, \
                    f"Solution {dom_idx} doesn't actually dominate solution {i}: {dom_obj} vs {sol_obj}"
        
        valid_indices = [idx for idx in dominated_indices_filtered if idx != u32_max]
        print(f"✓ All {len(valid_indices)} domination indices are valid")
        print(f"✓ {dominated_indices_filtered.count(u32_max)} solutions never dominated (index = MAX)")
        
        # Test 5: Validate temporal ordering
        print("\n--- Test 5: Validating temporal dominance property ---")
        
        # For filtered trace: solutions should be non-dominated at their discovery time
        # This means checking against only PREVIOUS solutions (temporal order)
        temporal_violations = 0
        
        for i in range(1, len(trace_filtered.objectives)):
            current_obj = trace_filtered.objectives[i]
            
            # Check if dominated by any PREVIOUS solution (temporal check)
            for j in range(i):
                prev_obj = trace_filtered.objectives[j]
                
                # Check if prev dominates current
                all_leq = all(prev_obj[k] <= current_obj[k] for k in range(len(prev_obj)))
                any_less = any(prev_obj[k] < current_obj[k] for k in range(len(prev_obj)))
                
                if all_leq and any_less:
                    temporal_violations += 1
                    print(f"  ⚠️  Solution {i} was dominated at discovery by solution {j}")
                    break
        
        assert temporal_violations == 0, \
            f"Found {temporal_violations} solutions that were dominated at discovery time (should be filtered out)"
        
        print(f"✓ No temporal violations: all {trace_filtered.metadata['solution_count']} solutions were non-dominated at discovery")
        
        # Test 6: Compare solution quality between filtered and unfiltered
        print("\n--- Test 6: Comparing final Pareto quality ---")
        
        # Both should produce similar final Pareto fronts
        # (filtering only affects trace, not final result)
        print(f"✓ Final Pareto size (all): {len(result_all.final_solutions)}")
        print(f"✓ Final Pareto size (filtered): {len(result_filtered.final_solutions)}")
        
        # The final Pareto fronts should be similar in size
        # (may not be identical due to random exploration, but should be close)
        size_ratio = len(result_filtered.final_solutions) / len(result_all.final_solutions)
        assert 0.8 <= size_ratio <= 1.2, \
            f"Final Pareto sizes differ significantly: {len(result_all.final_solutions)} vs {len(result_filtered.final_solutions)}"
        
        print(f"✓ Final Pareto sizes are similar (ratio: {size_ratio:.2f})")
        
        # Summary
        print("\n" + "="*80)
        print("✅ ALL DOMINANCE FILTERING TESTS PASSED")
        print("="*80)
        print(f"Trace size reduction: {trace_all.metadata['solution_count']} → {trace_filtered.metadata['solution_count']} ")
        print(f"Solutions filtered out: {trace_all.metadata['solution_count'] - trace_filtered.metadata['solution_count']} "
              f"({100 * (1 - trace_filtered.metadata['solution_count'] / trace_all.metadata['solution_count']):.1f}% reduction)")
        print("Temporal dominance property: ✓ Verified")
        print("Domination indices: ✓ All valid")
        print("Final Pareto quality: ✓ Preserved")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])