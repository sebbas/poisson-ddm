# poisson-ddm

### System setup
Create a new conda environment based on given file:

`cd poisson-ddm/src/`

`conda env create -f ../psn.yml`

`conda activate psn`

### Data generation

Generate a dataset with e.g. 1000 samples:

`python generate_Poisson_data.py -f ../data/psn -n 1000`

### Training

Train the homogeneous Poisson equation the dataset for 100 epochs and batchsize 64:

`python gfnet.py -n 1000 -e 100 -f ../data/psn_32_1000.h5 -b 64 -eq 0 -bc 0`

Use `-eq 1 -bc 1` for inhomogeneous Laplace and  `-eq 2 -bc 1` for inhomogeneous Poisson.

### Predicting

Predict 5 frames from the test dataset:

`python gfnet.py -n 1000 -e 100 -f ../data/psn_32_1000.h5 -b 64 -eq 0 -bc 0 -t 0 -p 5`

Note: `-t 0` just disables the training step.

### Loss plots

Adjust modelname in plotting script (file from command below) and plot loss and MAE curves:

`python plot_Poisson_model.py`
