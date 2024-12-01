#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# CSV header
echo "selector,test_suite_cnt,selection_cnt,time_to_initialize,time_to_select_tests,time_to_fault_ratio,fault_to_selection_ratio,diversity" > results/evaluation_results.csv

# Function to run evaluator and process output
run_evaluator() {
    local selector_name=$1
    echo "Running evaluator for $selector_name..."
    docker run --rm --name evaluator-container --network host -t evaluator-image -u localhost:4545 | tee "results/${selector_name}_output.txt"
    
    # Extract metrics from the output
    local metrics=$(grep "EvaluationReport" "results/${selector_name}_output.txt" | sed 's/[()]//g' | cut -d'=' -f2-)
    if [ ! -z "$metrics" ]; then
        echo "${selector_name},${metrics}" >> results/evaluation_results.csv
    fi
    
    # Wait a bit before next evaluation
    sleep 5
}

# Clean up function
cleanup() {
    echo "Cleaning up containers..."
    docker ps -aq | xargs -r docker stop
    docker ps -aq | xargs -r docker rm
    echo "Cleanup complete"
}

# Initial cleanup
echo "Performing initial cleanup..."
cleanup

# Trap for cleanup on script exit or interrupt
trap cleanup EXIT INT TERM

# Build evaluator image first
echo "Building evaluator image..."
cd evaluator
docker build -t evaluator-image .
cd ..

# Function to wait for port to be free
wait_for_port_free() {
    while lsof -i :4545 > /dev/null 2>&1; do
        echo "Waiting for port 4545 to be free..."
        sleep 2
    done
}

# Function to run a selector
run_selector() {
    local name=$1
    local build_dir=$2
    local extra_args=$3

    echo "Running $name experiment..."
    
    # Wait for port to be free
    wait_for_port_free
    
    # Build and run
    cd "$build_dir"
    docker build -t "$name" .
    docker run --rm --name "$name" -d --network host $extra_args "$name" -p 4545
    
    # Wait for service to start
    sleep 15
    
    cd ../..
    run_evaluator "$name"
    
    # Stop the container
    docker stop "$name"
    
    # Additional wait to ensure cleanup
    sleep 5
}

# 1. Sample Tool
run_selector "sample-selector" "tools/sample_tool" ""

# 2. ML Selector
run_selector "ml-selector" "tools/ml_selector" ""

# 3. Transformer Selector
run_selector "transformer-selector" "tools/transformer_selector" "--privileged --gpus all"

# 4. Graph Selector
run_selector "graph-selector" "tools/graph_selector" "--privileged --gpus all"

# 5. Curvature Selector
run_selector "curvature-selector" "tools/curvature_selector" ""

echo "All experiments completed. Results saved in results/evaluation_results.csv"
echo "Results directory contains individual output files for each selector"

# Print final results
echo "Final Results:"
cat results/evaluation_results.csv

# Final cleanup
cleanup