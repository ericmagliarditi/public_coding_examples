#!/bin/bash

# Initialize Modules
source /etc/profile

# Load Python MOdule
module load anaconda/2020a

#pip install the setup
pip install --user -e .

#Call Script
python bin/train.py --data-dir ../Data/training_data \
	--start-new-model 1 \
	--final-model-name adam_full_run_april_30 --model-paths ../Data/models \
	--epoch-loss-dir ../Data/ --input-image-size 612  \
	--label-size 540 --batch-size 8 --epochs 100 \
	--learning-rate 0.001 --checkpoint 20 \
	--training-split 0.9 --momentum 0.9