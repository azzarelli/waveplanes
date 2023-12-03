# Code Submission for *WavePlanes: A compact Wavelet representation for Dynamic Neural Radiance Fields*

## Setup 

We recommend using a conda environment (a high-memory GPU is not required). This should be run on a Linux OS, however we provide details below to modify the code and run on Windows. We consider this a branch off the [K-Planes](https://github.com/sarafridov/K-Planes) repository and follows the same installation process, so K-Planes Github issues may also prove helpful for debugging and more.

All data sets are freely avaliable online.

Configuration files are provided in `plenoxels/configs/final/[D-NeRF/DyNeRF/LLFF].py`.

We make minor modifications to the [pytorch_wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/readme.html) library to work with [native Pytorch mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) (using `fp16`). The modifications consist of two lines in `pytorch_wavelets_/dwt/lowlevel.py` which cast half-precision floats to regular floats. To use the modified library unzip `pytorch_wavelets_.zip` and place the contents in the base folder (file management discussed below).

We use [wandb](https://wandb.ai/) to track our results. This isn't necessary and can be 'turned off' using the command `wandb disabled`.

Finally after training and rendering all results can be found in the log folder.

*Note: We have not modified or deleted functionality for the K-Planes field and K-Planes proposal networks. To run this model, please change the configurations for K-Planes hyper-parameters and use set the `use_wavelet_field` and `use_wavelet_psn` options to `False`.*

## Commands

In `run.sh` we provide various ways to run the code. The main way is to train a model is to use the following command:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py option=[choice]
```

To render a model the following command can be used:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/log/folder --render-only expname=[experiment name]
```

To validate a model the following command can be used:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/log/folder --validate-only expname=[experiment name]
```

To decompose a model into static and dynamic components the following command can be used:
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py --log-dir path/to/log/folder --spacetime-only expname=[experiment name]
```

Note: to generate IST weights for the DyNeRF dataset, we recomend training a single iteration with 8x downsampling. Then run as usual with 2x down sampling.

## Compression

We compress our model after training. This can be done using the `--validate-only` flag which saves and loads the main field to the `fields/` folder. Note that the current implementation only supports saving one compressed model at a time. If you run the validation again it will overwrite any previously compressed models in the save folder.

The default is LZMA however alternatives are avaliable by using the `compression:[BZ2, LZMA, pickle, GZIP]` option in the `grid_config` variable in the config file.

## File Management

The main folder should contain:

1. data (place data sets in here)
2. fields (temporary folder for compressed models)
3. logs (results + uncompressed models)
4. plenoxels (code)
5. pytorch_wavelets_

Note: Depending on the file structure of imported data sets the relevant configuration files may need to be changed.


## How to: Windows

To modify the code to run on Windows we first change `plenoxels/main.py:Line 25` to `gpu = 0`. This expects a single GPU with id 0. If you have other GPUs change the line to match the id. The other modification is to comment out any line that relies on the `resource` package (e.g. `plenoxels/datasets/dataloading.py`:Line 3 & 16-17).

Note that it is likely the SSIM functionality will throw Warnings (not relevant to our implementation). These can be turned off by instantiating the [right code prior to running](https://stackoverflow.com/questions/879173/how-to-ignore-deprecation-warnings-in-python).