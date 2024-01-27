#!/bin/bash
python triforce.py train --parallel 10 --output /training &
TRAIN_PID=$!
tensorboard --logdir=/training --bind_all &
TENSORBOARDPID=$!

wait $TRAIN_PID
kill $TENSORBOARDPID
