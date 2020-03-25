#!/bin/sh

# Initialize Modules
source /etc/profile

# Load Python MOdule
module load anaconda3-2019b

#pip install the setup
# pip install --user -e .

#Call Script
python bin/run_ml_project.py --data-dir ../Data --final-model-name ADAM_Jan_6_LR_0.2 --model-paths ../Data/models --epoch-loss-dir ../Data/ --input-image-size 612  --label-size 500 --batch-size 8 --epochs 200 --learning-rate 0.2 --checkpoint 100