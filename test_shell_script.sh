#!/usr/bin/env bash
#$ -N tets_arg_pars  # name of the experiment: set consistent base name for output and error file (allows for easy deletion alias)
#$ -l cuda=1   # remove this line when no GPU is needed!
#$ -q all.q       # don't fill the qlogin queue
#$ -cwd          # start processes in current directory
#$ -V              # provide environment variables
#$ -M louisedqne@gmail.com

echo “Starting test_arg_pars”
python3 test_parser.py --count 3 --name Louise # here you perform any commands to start your program
