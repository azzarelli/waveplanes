# Select config file and define an experiment name
CONFIG=plenoxels/configs/final/LLFF/llff.py
EXPNAME='w_fern'

CONFIG=plenoxels/configs/final/D-NeRF/trex.py
EXPNAME='w_trex'

CONFIG=plenoxels/configs/final/DyNeRF/dynerf.py
EXPNAME='w_cutbeef'

# Training:
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME fusion=ZMM
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG  expname=$EXPNAME fusion=ZAM

# Validation:
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG --log-dir './logs/LLFF/'$EXPNAME --validate-only expname=$EXPNAME

# Rendering:
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG --log-dir './logs/LLFF/'$EXPNAME --render-only expname=$EXPNAME

# Decompose Space-time:
PYTHONPATH='.' python plenoxels/main.py --config-path $CONFIG --log-dir './logs/DyNeRF/'$EXPNAME --spacetime-only expname=$EXPNAME
