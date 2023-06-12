# poisson-ddm

### Create a new conda environment based on given file:
`cd poisson-ddm/src/`

`conda env create -f ../psn.yml`

### Generate a dataset with e.g. 1000 samples
`python generate_Poisson_data.py -f ../data/psn -n 1000`

### Train with the dataset for 100 epochs and batchsize 64
`python gfnet.py -n 1000 -e 100 -f ../data/psn_32_1000.h5 -b 64`

### Adjust modelname in plotting script (file from command below) and plot loss and MAE curves:
`python plot_Poisson_model.py`
