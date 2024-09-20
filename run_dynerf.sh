SCENE='flame_salmon'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"

SCENE='coffee_martini'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"

SCENE='cut_roasted_beef'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"

SCENE='sear_steak'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"

SCENE='cook_spinach'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"

SCENE='flame_steak'
python gui.py -s /media/xi22005/DATA/dynerf/${SCENE} --port 6017 --expname "dynerf/${SCENE}_MUL" --configs arguments/dynerf/${SCENE}.py --gui
python render.py --model_path "output/dynerf/${SCENE}_MUL/"  --skip_train --configs arguments/dynerf/${SCENE}.py
python metrics.py --model_path "output/dynerf/${SCENE}_MUL"