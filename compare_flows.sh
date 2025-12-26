#!/bin/bash

echo "==================================================================="
echo "Running Rust GPBA-A with detailed logging..."
echo "==================================================================="
cd /home/vhlushchenko/sims-hybrid-algs/augmecon-rs
cargo test test_main_loop_with_hardcoded_inputs --release -- --nocapture 2>&1 | \
    grep -E "(ITERATION|ef_array =|NEW solution|DUPLICATE|INFEASIBLE)" | \
    head -100 > /tmp/rust_flow.txt

echo ""
echo "==================================================================="
echo "Running Python GPBA-A with detailed logging..."
echo "==================================================================="
cd /home/vhlushchenko/sims-hybrid-algs/sims-solvers
uv run pytest tests/test_gpba_phases.py::TestMainLoop::test_main_loop_with_hardcoded_inputs -v -s 2>&1 | \
    grep -E "(Iteration [0-9]+|ef_array|Found solution [0-9]+:|MAIN LOOP: After solve)" | \
    head -100 > /tmp/python_flow.txt

echo ""
echo "==================================================================="
echo "Comparing first 50 iterations..."
echo "==================================================================="
echo ""
echo "RUST:"
head -50 /tmp/rust_flow.txt
echo ""
echo "==================================================================="
echo "PYTHON:"
head -50 /tmp/python_flow.txt
echo ""
echo "==================================================================="
echo "Solution counts:"
echo "Rust unique solutions:"
grep "NEW solution" /tmp/rust_flow.txt | wc -l
echo "Python solutions:"
grep "Found solution [0-9]" /tmp/python_flow.txt | wc -l
