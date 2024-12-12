CONFIG=plenoxels/configs/final/DyNeRF/dynerf.py
EXPNAME='flame_steak_test'
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME fusion=HP
