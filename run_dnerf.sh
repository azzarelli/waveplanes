
CONFIG=plenoxels/configs/dnerf_w3/
EXPNAME='W_LRLoss-2-psnr'
SCENES=('lego' 'mutant' 'bouncingballs' 'hook' 'hellwarrior' 'standup' 'jumpingjacks')
# SCENES=('trex')


for SCENE in "${SCENES[@]}"; do
    PYTHONPATH='.' python plenoxels/main.py --config-path "$CONFIG$SCENE.py"  expname="${EXPNAME}_${SCENE}"
    PYTHONPATH='.' python plenoxels/main.py --config-path "$CONFIG$SCENE.py" --log-dir './logs/'$SCENE'/'"${EXPNAME}_${SCENE}" --validate-only expname="${EXPNAME}_${SCENE}"
done 