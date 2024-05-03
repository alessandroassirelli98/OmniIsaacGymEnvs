#!/bin/bash

echo '

 ------ Running run1 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=False +nn_type=SharedNetworks
echo '

 ------ Running run2 -------

'
python scripts/skrl/diana_tekken_ppofd.py +pretrain=True +nn_type=SharedNetworks
