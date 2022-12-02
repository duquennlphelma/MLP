#!/usr/bin/env bash
#$ -N pml_O7_train_50_epochs  # name of the experiment: set consistent base name for output and error file (allows for easy deletion alias)
#$ -l cuda=1   # remove this line when no GPU is needed!
#$ -q all.q       # don't fill the qlogin queue
#$ -cwd          # start processes in current directory
#$ -V              # provide environment variables
#$ -M louisedqne@gmail.com

echo “Staring Real NVP script”
python train.py  # here you perform any commands to start your program
