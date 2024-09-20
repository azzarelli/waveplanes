
SCENE='lego'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"

SCENE='hellwarrior'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"

SCENE='hook'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"

SCENE='jumpingjacks'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"

SCENE='mutant'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"

SCENE='standup'
python gui.py -s /home/xi22005/DATA/dnerf/${SCENE} --port 6017 --expname "dnerf/${SCENE}_MUL" --configs arguments/dnerf/${SCENE}.py --gui
python render.py --model_path "output/dnerf/${SCENE}_MUL/"  --skip_train --configs arguments/dnerf/${SCENE}.py
python metrics.py --model_path "output/dnerf/${SCENE}_MUL"