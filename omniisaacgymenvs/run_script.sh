#!/bin/bash

echo 'Running run1'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4
echo 'Running run2'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run3'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4
echo 'Running run4'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler
