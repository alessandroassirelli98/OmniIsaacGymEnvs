#!/bin/bash

echo 'Running run1'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4
echo 'Running run2'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run3'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4
echo 'Running run4'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run5'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +lambda_0=1.0
echo 'Running run6'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +lambda_0=1.0
