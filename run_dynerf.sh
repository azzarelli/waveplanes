#EXP='W-4DGS-OCDIM32'
#
#SCENES=("cook_spinach" "cut_roasted_beef" "flame_steak" "sear_steak")
##SCENES=("flame_salmon")
##SCENES=("coffee_martini")
#
#
#for SCENE in "${SCENES[@]}"; do
#  python gui.py -s /data/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py --gui
#  python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#  python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"
#done

EXP='FSa_TEST'
SCENE="flame_salmon"
python render.py --model_path "output/dynerf/${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py

