#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SCRIPT_PATH="./main.py"

SEEDS=(88 99 66 999)
NGFS=(128 256 384)
BATCH_SIZES=(512 1024)
EXP_ID=1

echo "================================================================"
echo "Starting automated experiments at $(date)"
echo "Combinations: 4 Seeds x 2 NGFs x 3 BatchSizes = 24 Experiments"
echo "================================================================"

for ngf in "${NGFS[@]}"; do
    for b in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            echo "----------------------------------------------------"
            echo "Running Experiment $EXP_ID: ngf=$ngf, b=$b, seed=$seed"
            echo "Start time: $(date)"
            python "$SCRIPT_PATH" \
                --exp "$EXP_ID" \
                --size 600 \
                --num_d 160 \
                --seed "$seed" \
                --lr 0.0005 \
                --ngf "$ngf" \
                --beta1 0.9 \
                --b "$b" \
                --threshold 0.5 \
                --device cuda:1

            # Check whether the previous command was successfully executed
            if [ $? -eq 0 ]; then
                echo "Experiment $EXP_ID finished successfully."
            else
                echo "Experiment $EXP_ID failed!"
            fi
            # experiment number increments automatically
            EXP_ID=$((EXP_ID + 1))
            
        done
    done
done

echo "================================================================"
echo "All experiments completed at $(date)"
echo "================================================================"