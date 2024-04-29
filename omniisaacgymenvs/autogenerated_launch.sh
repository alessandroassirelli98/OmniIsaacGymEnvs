#!/bin/bash

echo 'Running run1'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4
echo 'Running run2'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run3'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4
echo 'Running run4'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4
echo 'Running run5'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4
echo 'Running run6'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4
echo 'Running run7'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run8'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run9'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run10'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler
echo 'Running run11'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +lambda_0=1.0 +lambda_1=0.999
echo 'Running run12'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +lambda_0=1.0 +lambda_1=0.999
echo 'Running run13'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +lambda_0=1.0 +lambda_1=0.9999
echo 'Running run14'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +lambda_0=1.0 +lambda_1=0.9999
echo 'Running run15'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +lambda_0=0.5 +lambda_1=0.9999
echo 'Running run16'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +lambda_0=0.5 +lambda_1=0.9999
echo 'Running run17'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +lambda_0=0.5 +lambda_1=0.99
echo 'Running run18'
python scripts/skrl/diana_tekken_PPOFD.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +lambda_0=0.5 +lambda_1=0.99
