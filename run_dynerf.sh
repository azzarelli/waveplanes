EXP='rotator_3'

SCENE='cut_roasted_beef'
#python scripts/preprocess_dynerf.py --datadir /media/xi22005/DATA/dynerf/${SCENE}
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"
##
#SCENE='cook_spinach'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"
#
#SCENE='sear_steak'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"
#
#SCENE='flame_steak'
#python scripts/preprocess_dynerf.py --datadir /media/xi22005/DATA/dynerf/${SCENE}
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"


#SCENE='coffee_martini'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_MUL"
#
#SCENE='cut_roasted_beef'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_MUL"
#
#SCENE='sear_steak'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_MUL"
#
#SCENE='cook_spinach'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_MUL"
#
#SCENE='flame_steak'
#python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
#python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
#python metrics.py --model_path "output/dynerf/${SCENE}_MUL"