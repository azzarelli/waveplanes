
CONFIG=plenoxels/configs/dynerf/
EXPNAME='120k'
SCENES=('flame_steak' 'cut_roasted_beef' 'sear_steak' 'coffee_martini' 'flame_salmon')
SCENES=('flame_steak')

# SCENES=('trex')


for SCENE in "${SCENES[@]}"; do
    PYTHONPATH='.' python plenoxels/main.py --config-path "$CONFIG$SCENE.py"  expname="${EXPNAME}_${SCENE}"
    PYTHONPATH='.' python plenoxels/main.py --config-path "$CONFIG$SCENE.py" --log-dir './logs/'$SCENE'/'"${EXPNAME}_${SCENE}" --validate-only expname="${EXPNAME}_${SCENE}"
done 