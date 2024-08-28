# CONFIG=plenoxels/configs/final/D-NeRF/standup_w3.py
# EXPNAME='standup_w3_60k'
# PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME

# CONFIG=plenoxels/configs/final/D-NeRF/standup_75timeres.py
# EXPNAME='standup_75timeres'
# PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME

CONFIG=plenoxels/configs/final/D-NeRF/standup_lr_w360k.py
EXPNAME='standup_lr_w3_60k'
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME