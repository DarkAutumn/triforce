#!/bin/bash
python triforce.py train --parallel 10 --output /output/training &
TRAIN_PID=$!
tensorboard --logdir=/output/training --bind_all &
TENSORBOARDPID=$!

wait $TRAIN_PID
kill $TENSORBOARDPID
