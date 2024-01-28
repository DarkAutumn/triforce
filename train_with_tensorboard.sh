#!/bin/bash

# for use with docker

python triforce.py train --parallel 16 --output /training &
TRAIN_PID=$!
tensorboard --logdir=/training --bind_all &
TENSORBOARDPID=$!

wait $TRAIN_PID
kill $TENSORBOARDPID
