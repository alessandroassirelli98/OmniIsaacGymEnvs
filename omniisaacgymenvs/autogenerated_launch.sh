#!/bin/bash

echo '

 ------ Running run1 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +nn_type=SharedNetworks
echo '

 ------ Running run2 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SharedNetworks
echo '

 ------ Running run3 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SharedNetworks
echo '

 ------ Running run4 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SharedNetworks
echo '

 ------ Running run5 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +nn_type=SeparateNetworks
echo '

 ------ Running run6 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SeparateNetworks
echo '

 ------ Running run7 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SeparateNetworks
echo '

 ------ Running run8 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SeparateNetworks
echo '

 ------ Running run9 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +nn_type=SharedNetworks
echo '

 ------ Running run10 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SharedNetworks
echo '

 ------ Running run11 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SharedNetworks
echo '

 ------ Running run12 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SharedNetworks
echo '

 ------ Running run13 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +nn_type=SeparateNetworks
echo '

 ------ Running run14 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SeparateNetworks
echo '

 ------ Running run15 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SeparateNetworks
echo '

 ------ Running run16 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SeparateNetworks
echo '

 ------ Running run17 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run18 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run19 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run20 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run21 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run22 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run23 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run24 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run25 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run26 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run27 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run28 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SharedNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run29 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run30 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run31 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +nn_type=SeparateNetworks +entropy_loss_scale=0.1
echo '

 ------ Running run32 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +mini_batches=4 +value_preprocessor=RunningStandardScaler +reward_shaper=True +learning_rate_scheduler=KLAdaptiveRL +nn_type=SeparateNetworks +entropy_loss_scale=0.1
