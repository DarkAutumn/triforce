#!/bin/bash

# for use with docker

mkdir -p /training/current
python triforce.py train --parallel 16 --output /training/current &
TRAIN_PID=$!
tensorboard --logdir=/training --bind_all &
TENSORBOARDPID=$!

wait $TRAIN_PID
kill $TENSORBOARDPID
