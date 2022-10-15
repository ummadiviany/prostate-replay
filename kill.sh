# !/bin/bash

pkill -f train.py
kill $(ps -ef | grep train.py)