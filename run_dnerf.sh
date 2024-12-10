#
#EXP='W-4DGS'
##SCENES=("trex" "standup" "mutant" "lego" "jumpingjacks" "bouncingballs" "hook" "hellwarrior")
#SCENES=("lego")
#for SCENE in "${SCENES[@]}"; do
#  python gui.py -s /data/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
#  python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
#  python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"
#done


EXP='W-4DGS'
 SCENES=("trex" "standup" "mutant" "lego" "jumpingjacks" "bouncingballs" "hook" "hellwarrior")
#SCENES=("lego")
for SCENE in "${SCENES[@]}"; do
#  python gui.py -s /data/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
  python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
#  python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"
done