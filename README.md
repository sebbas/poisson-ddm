# poisson-ddm

### System setup

```bash
cd poisson-ddm/
conda env create -f psn.yml
conda activate psn
```

### Data generation

```
python generate_Poisson_data.py -f ../data/psn -n 20000
```

Generates a `hdf5` dataset with 20.000 samples. One sample consists of 32x32 grids for components `a` (coefficient), `f` (right-hand side), `bc` (boundary condition), and `p` (solution) of the variable coefficient Poisson equation:
```math
\nabla \cdot (a \nabla p(x,y)) = f(x,y), \qquad (x,y) \in \Omega \\
p(x,y) = bc(x,y) \qquad (x,y) \in \partial\Omega
```

### Training

```
python gfnet.py -n 20000 -e 500 -f ../data/psn_32_20000.h5 -b 64 -eq 2 -l 2
```
Trains the inhomogeneous Poisson equation with 20.000 samples (90/10 train/valid split) for 500 epochs with a batchsize of 64 samples.
Alternatively, the inhomogeneous Laplace equation (`f=0`) or the homogeneous Poisson (`bc=0`) can be trained with options `-eq 1` and `-eq 0`

### Predictions

```
python generate_Poisson_data.py -f ../data/test -n 100
python gfnet.py -n 100 -f ../data/test_32_100.h5 -eq 2 -l 2 -t 0 -p 5
```
Generates a new dataset to test on. Then predicts the first 5 frames (`-p 5`) from this dataset, plots results and saves them in `../img/`.
Calling `gfnet` with option `-t 0`  disables the training step and restores the model that was trained previously.

###  Architectures

`gfnet.py` contains 3 U-Net implementations that can be loaded with option `-l`.
-  Simplified U-Net (`-l 0`)
-  Full U-Net with Dropout (`-l 1`)
-  Full U-Net with Batch Normalization (`-l 2`)

New architectures can be created similarly to the existing ones (the model layers are wrapped in a list of strings).


