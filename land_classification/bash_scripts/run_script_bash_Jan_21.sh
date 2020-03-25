#!/bin/sh

# Initialize Modules
source /etc/profile

# Load Python MOdule
module load anaconda3-2019b

#pip install the setup
pip install --user -e .

#Call Script
python bin/run_ml_project.py --data-dir ../Data/training_data \
	--start-new-model 0 --model-to-load jan_9_SGD \
	--final-model-name jan_21_SGD --model-paths ../Data/models \
	--epoch-loss-dir ../Data/ --input-image-size 612  \
	--label-size 420 --batch-size 8 --epochs 200 \
	--learning-rate 0.005 --checkpoint 100 \
	--training-split 0.95 --momentum 0.9