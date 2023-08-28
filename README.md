# poisson-ddm

A neural network that solves the variable coefficient Poisson equation.

## System setup

```bash
cd poisson-ddm/
conda env create -f psn.yml
conda activate psn
```

## Training data generation

```
python generate_Poisson_data.py -f ../data/train -n 20000
```

Generates a `hdf5` dataset with 20.000 samples. One sample consists of 32x32 grids for components `a` (coefficient), `f` (right-hand side), `g` (boundary condition), and `p` (solution) of the variable coefficient Poisson equation:

$$
\begin{aligned}
\nabla \cdot (a \nabla p(x,y)) &= f(x,y), \qquad (x,y) \in \Omega \\
p(x,y) &= g(x,y), \qquad (x,y) \in \partial\Omega
\end{aligned}
$$

![alt text](https://dl.dropboxusercontent.com/scl/fi/ev5ndojbqxb7mzluymyy2/train.gif?rlkey=0ln2388j9fyia6388mt9aj1p7&dl=0 "Model inputs (bc, a, f), labels (p)")
***Training data:** Selection of Poisson equation instances for model training*

## Training

```
python gfnet.py -n 20000 -e 500 -f ../data/train_32_20000.h5 -b 64 -eq 2 -l 2
```
Trains the inhomogeneous Poisson equation with 20.000 samples (90/10 train/valid split) for 500 epochs with a batchsize of 64 samples.
Alternatively, the Laplace (`f=0`) or the Poisson equation with zero-BC (`bc=0`) can be trained with options `-eq 1` and `-eq 0`.

## Test data generation

```
python generate_Poisson_data.py -f ../data/test -n 100
```
Generates a new dataset to test on. Option `-p` generates a smooth dataset with additional (also previously unseen) samples between randomly generated Poisson instances.

![alt text](https://dl.dropboxusercontent.com/scl/fi/vb337bl0f3i7uoy7s8ne3/infer.gif?rlkey=ht0br2uz0st8r8k2lye9i8xzx&dl=0 "Model inputs (bc, a, f), labels (p)")
***Test data:** Selection of Poisson equation instances for model testing*

## Predictions

### Inhomogeneous Poisson
```
python gfnet.py -n 100 -f ../data/test_32_100.h5 -eq 2 -l 2 -t 0 -p 5
```
Predicts the first 5 frames (`-p 5`) from the dataset, plots results and saves them in `../img/`.
Calling `gfnet` with option `-t 0`  disables the training step and restores the model that was trained previously.

![alt text](https://dl.dropboxusercontent.com/scl/fi/0ygjnx1p9bd1v8lfeaeak/pred.gif?rlkey=sjv0h9vmfi779bgebz08a114s&dl=0 "Labels (p), predictions (phat), error (|p-phat|), error metrics")
***Ground truth vs. predictions:** Model trained on the inhomogeneous Poisson equation*

### Poisson zero-BC + Laplace

```
python gfnet.py -n 20000 -e 500 -f ../data/train_32_20000.h5 -b 64 -eq 0 -l 2
python gfnet.py -n 20000 -e 500 -f ../data/train_32_20000.h5 -b 64 -eq 1 -l 2
```
Separating the inhomogeneous Poisson equation into models for the Poisson zero-BC and Laplace equation and summing their predictions improves RMSE and MAE.

![alt text](https://dl.dropboxusercontent.com/scl/fi/88eovui51wj2q4g7jt6qg/compare.gif?rlkey=ntmfa8k256ohxzzzkfozzy8j7&dl=0 "Predictions (phat), error (|p-phat|), error metrics")
***Separating predictions into 2 models:** (1st row) Predicting solution to the Poisson equation directly. (2nd row): Sum of predictions from the Poisson zero-BC and Laplace models. Results in improved MAE and RMSE.*


##  Architectures

`gfnet.py` contains 3 U-Net implementations that can be loaded with option `-l`.
-  Simplified U-Net (`-l 0`)
-  Full U-Net with Dropout (`-l 1`)
-  Full U-Net with Batch Normalization (`-l 2`)

## Benchmarks

```
python Poisson_benchmark.py -f ../data/test_32_100.h5 -n 100 -b 2
```
Solves `n` instances from the test dataset numerically and logs times for all equations. Default solving backend is [`pyamgx`](https://github.com/shwina/pyamgx) (GPU). Alternatively, solving Poisson with the CPU is possible with [`pyamg`](https://github.com/pyamg/pyamg) (`-b 1`) or [`scipy.sparse.linalg.spsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) (`-b 0`).

This utility script can be used to compare the inference time (performance / speed) of the models against a state-of-the-art numerical solver.
