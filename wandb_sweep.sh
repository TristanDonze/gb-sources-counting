#!/bin/bash

# Script to run wandb hyperparameter sweep
# Usage: ./run_sweep.sh <project_name> <count> [sweep_id]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <project_name> <count> [sweep_id]"
    exit 1
fi

PROJECT=$1
COUNT=$2
SWEEP_ID=$3

if [ -z "$SWEEP_ID" ]; then
    echo "Creating new wandb sweep for project: $PROJECT"
    
    # Create the sweep
    SWEEP_ID=$(wandb sweep sweep.yaml --project $PROJECT 2>&1 | grep "Run sweep agent with:" | grep -oP 'wandb agent \K\S+')    
    
    if [ -z "$SWEEP_ID" ]; then
        echo "Failed to create sweep. Make sure wandb is logged in."
        exit 1
    fi
    
    echo "Created sweep with ID: $SWEEP_ID"
else
    echo "Running with existing sweep ID: $SWEEP_ID"
fi

echo "Starting wandb agent for sweep: $SWEEP_ID with count: $COUNT"
echo ""

# Run the wandb agent
# Note: W&B sweep agent will automatically inject hyperparameters through wandb.config
# and set WANDB_RUN_ID environment variable
wandb agent $SWEEP_ID \
    --count=$COUNT \
    --project=$PROJECT
