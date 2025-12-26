#!/bin/bash
# Benchmark GPBA complete pipeline test with different solvers
# Timeout: 6 minutes (allows overhead for 5 min internal timeout)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          GPBA Solver Benchmark Comparison                 ║"
echo "║          Timeout: 360 seconds per solver                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

TIMEOUT=360
TEST_NAME="test_complete_pipeline_end_to_end"

# Store results
declare -A results
declare -A times

# Function to run test with a solver
run_test() {
    local solver=$1
    local features=$2
    local solver_name=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing with ${solver_name} solver..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Modify test file to use this solver
    sed -i "s/Options::default()/Options::default().with_solver(Solver::${solver})/" tests/test_gpba_phases.rs
    
    # Run the test with timeout
    local start_time=$(date +%s.%N)
    timeout $TIMEOUT cargo test $TEST_NAME $features --release -- --nocapture > /tmp/gpba_test_output.txt 2>&1
    local exit_code=$?
    local end_time=$(date +%s.%N)
    
    # Restore original file
    sed -i "s/Options::default().with_solver(Solver::${solver})/Options::default()/" tests/test_gpba_phases.rs
    
    if [ $exit_code -eq 124 ]; then
        echo "❌ TIMEOUT after ${TIMEOUT}s"
        results[$solver_name]="TIMEOUT"
        times[$solver_name]=">${TIMEOUT}"
    elif [ $exit_code -ne 0 ]; then
        echo "❌ FAILED (exit code: $exit_code)"
        # Show error
        tail -30 /tmp/gpba_test_output.txt
        results[$solver_name]="FAILED"
        times[$solver_name]="N/A"
    else
        # Extract time and solutions from output
        local elapsed=$(echo "$end_time - $start_time" | bc)
        local solutions=$(grep "Solutions found:" /tmp/gpba_test_output.txt | tail -1 | awk '{print $NF}')
        local non_dominated=$(grep "Non-dominated:" /tmp/gpba_test_output.txt | tail -1 | awk '{print $NF}')
        
        echo "✅ SUCCESS"
        echo "   Time: ${elapsed}s"
        echo "   Solutions: ${solutions}"
        echo "   Non-dominated: ${non_dominated}"
        
        results[$solver_name]="$solutions"
        times[$solver_name]=$(printf "%.2f" $elapsed)
    fi
    echo ""
}

# Test Default solver (Gurobi) - skip if not available
echo "[1/3] Testing Default solver (Gurobi)..."
run_test "Default" "" "Default"

# Test CoinCbc solver
echo "[2/3] Testing CoinCbc solver..."
run_test "CoinCbc" "--features coin_cbc" "CoinCbc"

# Test HiGHS solver  
echo "[3/4] Testing HiGHS solver..."
run_test "HiGHS" "--features highs" "HiGHS"

# Test SCIP solver
echo "[4/4] Testing SCIP solver..."
run_test "SCIP" "--features scip" "SCIP"

# Print summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK RESULTS                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
printf "%-20s %12s %15s\n" "Solver" "Solutions" "Time (sec)"
echo "──────────────────────────────────────────────────────────────"

for solver in "Default" "CoinCbc" "HiGHS" "SCIP"; do
    printf "%-20s %12s %15s\n" "$solver" "${results[$solver]}" "${times[$solver]}"
done

# Find fastest solver
echo ""
fastest_solver=""
fastest_time=999999
for solver in "Default" "CoinCbc" "HiGHS" "SCIP"; do
    time_val=${times[$solver]}
    if [[ ! "$time_val" =~ [A-Za-z] ]] && [ ! -z "$time_val" ]; then
        if (( $(echo "$time_val < $fastest_time" | bc -l) )); then
            fastest_time=$time_val
            fastest_solver=$solver
        fi
    fi
done

if [ ! -z "$fastest_solver" ]; then
    echo "🏆 Fastest solver: $fastest_solver (${fastest_time}s)"
fi
echo ""
