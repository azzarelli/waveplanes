CONFIG=plenoxels/configs/final/D-NeRF/standup.py
EXPNAME='standup_test_bench'
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME fusion=ZAM
