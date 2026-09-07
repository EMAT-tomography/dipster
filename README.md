# DIP-STER
Deep Image Priors for Space Time Environment Reconstruction (DIP-STER) is a neural network that uses Deep Image Priors for machine learning using an architecture involving manifold learning and convolution nueral networks in order to determine the resolve a 4D(3D + time) series of electron tomography data during _in situ_ experiments.

## 1. Installation

```bash
# 1. GPU tomography stack (conda) -- match CUDA to your GPU.
#    RTX 5090 (Blackwell) needs CUDA 12.8+.
conda install -c pytorch -c nvidia -c conda-forge -c astra-toolbox -c aahendriksen \
    pytorch torchvision pytorch-cuda=12.8 tomosipo astra-toolbox=2.4

# 2. The package and its pip dependencies
pip install -e dipster-bare
```

## 2. Usage 

The main code is in `dipster_bare` (the name comes from previous version that included heavy dependencies to companion libraries!) and Jupyter notebooks with typical usage are included in `notebooks`.

All the notebooks will assume that a continuous tilt series is available as .tiff (but .mrc files can also be loaded with your preferred method) with angles and times metadata in a separate, 2-column .csv file.

We advise starting with `Train.ipynb` to confirm the code runs on your computer (friendlier test notebooks, additional doc and test data will follow shortly) and get a feeling for the different hyperparameters. `Train_Hypertune.ipynb` follows the same structure but allows for runing an automatic hyperparameter study with [Optuna](https://optuna.org).

In addition, the `Moving_window_tomo.ipynb` notebook includes routines for moving window TV, EM and SIRT reconstructions, try it yourself!

## 3. License 
This code is licensed under GNU general public license version 3.0.

## 4. Citations and acknowledgments
See the preprint at: https://arxiv.org/abs/2603.29462

DIP-STER draws inspiration from many great works including but not limited to:

[Time Dependent Deep Image Priors](https://github.com/jaejun-yoo/TDDIP)\
[Tomosipo](https://github.com/cicwi/tomosipo) and its [algorithms](https://github.com/ahendriksen/ts_algorithms)

Thank you!

## 5. Contributors
Timothy Craig - tim.craig@uantwerpen.be\
Adrien Moncomble - adrien.moncomble@uantwerpen.be\
Robin Girod - robin.girod@uantwerpen.be