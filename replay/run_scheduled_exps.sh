#!/bin/bash

# /home/amritesh/anaconda3/envs/torch/bin/python agps.py --order a,b,c,d --device cpu --epochs 10 --replay >> /home/amritesh/anaconda3/replay/outputs/testout.txt && echo "Done with 1st experiment"

# /home/amritesh/anaconda3/envs/torch/bin/python agps.py --order a,b,c,d --device cuda:0 --epochs 15 --no-replay >> /home/amritesh/anaconda3/replay/outputs/testout1.txt && echo "Done with 2nd experiment"

# /home/amritesh/anaconda3/envs/torch/bin/python agps.py --order a,b,c,d --device cuda:0 --epochs 15 --no-replay >> /home/amritesh/anaconda3/replay/outputs/testout3.txt && echo "Done with 3nd experiment"

# for arg in "$@"
# do 
# echo $arg
# done

# function run_exp {
#     echo "Running experiment in function"
# }

# run_exp