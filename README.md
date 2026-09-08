# DIP-STER
[Deep Image Priors for Space Time Environment Reconstruction (DIP-STER)](https://arxiv.org/abs/2603.29462) is a neural network framework that predicts a 4D (3D + time) series of electron tomography reconstructions from a single, continuous tilt series of a dynamically evolving specimen such as obtained during _in situ_ experiments or from beam damage.

<p align="center">
    <img src="Images/Star.gif" width="128">
    <img src="Images/Cube.gif" width="128">
    <img src="Images/Slice.gif" width="128">
</p>

DIP-STER works by using an implicit neural representation of the volume time series that *implicitely* regularizes for smoothness in time and along the **x** and **z** directions (assuming rotation around **y**). Coupled with a GRS-style tilt scheme that involves large tilt steps, these priors promote decoupling changes in the tilt series that originate from tilting from the actual sample dynamics.  

<p align="center">
    <img src="Images/Workflow.png" width="500">
</p>

The network takes in a 3-coordinates $(t_i, y_k, \theta_i)$ manifold, encodes it it via a FC net into a latent vector, and finally decodes it into a 2D orthoslice of the volume-time series. During training, this orthoslice is reprojected at angle $\theta_i$ and compared with the experimental (1D) projection at time $t_i$ and position $y_k$. 

<p align="center">
    <img src="Images/DIPSTER.png" width="500">
</p>

## 1. Installation

To be completed and further tested ...

```bash
# 1. CUDA-depended libraries, match CUDA version to your GPU
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

## 3. Citations and acknowledgments
If using DIP-STER or part of the code herein, please cite:  

> Craig, T. M.; Moncomble, A.; Kadu, A. A.; Vinnacombe-Willson, G. A.; Liz-Marzán, L. M.; Girod, R.; Bals, S. Continuous Three-Dimensional Imaging of Nanoscale Dynamics by in Situ Electron Tomography. **2026**. Preprint at https://doi.org/10.48550/arXiv.2603.29462

For functions defining ET-adapted GRS tilt schemes and other options for continuous tilting in TEM see:

> Craig, T. M.; Girod, R.; Vinnacombe-Willson, G.; Liz-Marzán, L. M.; Bals, S. Towards Continuous Time-Dependent Tomography: Implementation and Evaluation of Continuous Acquisition Schemes in Electron Tomography. *Ultramicroscopy* **2025**, 277, 114207. https://doi.org/10.1016/j.ultramic.2025.114207

---
DIP-STER draws inspiration from many great works including but not limited to:

- [Time Dependent Deep Image Priors](https://github.com/jaejun-yoo/TDDIP)
- [Tomosipo](https://github.com/cicwi/tomosipo) and its [algorithms](https://github.com/ahendriksen/ts_algorithms)

Thank you!

---
Interested in dynamic ET and implicit neural representation? See also from the community:
>Lim, C.; Casert, C.; McCray, A. R. C.; Lee, S.; Barnum, A.; Dionne, J.; Ophus, C. Missing Wedge Inpainting and Joint Alignment in Electron Tomography through Implicit Neural Representations. **2025**. Preprint at https://doi.org/10.48550/arXiv.2512.08113

> Chien, T.; Ophus, C.; Waller, L. Space-Time Implicit Neural Representations for Atomic Electron Tomography on Dynamic Samples. In *NeurIPS 2023 Workshop on Deep Learning and Inverse Problems*. **2023**.


## 4. Contributors
Timothy Craig - tim.craig@uantwerpen.be\
Adrien Moncomble - adrien.moncomble@uantwerpen.be\
Robin Girod - robin.girod@uantwerpen.be