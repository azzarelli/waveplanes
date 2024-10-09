
# SCENE='jumpingjacks'
# EXP='testRotator2'
# python gui.py -s /media/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
# python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
# python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"

SCENE='hellwarrior'
python gui.py -s /media/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"

SCENE='hook'
python gui.py -s /media/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"

SCENE='mutant'
python gui.py -s /media/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_${EXP}" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_${EXP}"

#SCENE='standup'
#python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
#python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
#python metrics.py --model_path "output/dnerf/${SCENE}_MUL"