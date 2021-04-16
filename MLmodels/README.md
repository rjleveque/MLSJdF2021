
# ML-Tsunami

## Requirements

The python scripts used to generate made use of the following python 
versions and packages
* ``anaconda3/2020.07``
* ``requests``
* ``tqdm``
* ``sklearn``
* ``pandas``
* ``tsfresh``
* ``matplotlib``
* ``torch==1.7.1+cu110``


## Dataset

``data`` subdirectory contains python scripts that prepare the data.

### Download data

Download and untar dataset (about 111 Mb)
```
python down_data.py
```
### Process data 

Prepare and process data for regression tasks with
```
python proc_data.py
```
shuffle training and test dataset, and nterpolates surface elevation 
data to time grid of size 256 and 

### Slurm scripts

Alternatively, edit and run the slurm script
```
sbatch run_data.s
```

## Support Vector Machine (SVM) / Random Forest Regression (RFR)

``scripts`` subdirectory contains python scripts that train and test
SVMs or RFRs and plots the results.

### Train and test SVM or RFR

Train and test SVMs using ``tsfresh`` features, SVMs without
the features, RFRs, respectively, by running 
```
python svm_train_test.py
python svm_raw_train_test.py
python rfr_train_test.py
```
Generate plots of the test results for each results above with
```
python svm_plot.py
python svm_raw_plot.py
python svm_rfr_plot.py
```

### Slurm scripts

Alternatively, edit and run the slurm script
```
sbatch run_svm_rfr.s
```

## Denoising Autoencoder (DAE) / Variational Autoencoder (VAE)

``scripts`` subdirectory contains python scripts that train and test
DAEs or VAEs and plots the results.

### Train DAE or VAE

Train autoencoders by
```
python dae_train.py
python vae_train.py
```
The script defaults to using a NVIDIA GPU for
training to boost performance. If the training on a CPU machine set
``AE.device = 'cpu'`` in the two scripts.

### Test DAE or VAE

Test autoencoders by running 
```
python dae_test.py
python vae_test.py
```
To test the autoencoders for realizations (non-KL realizations) 
outside of the dataset, run
```
python dae_test_nKL.py
python vae_test_nKL.py
```

### Plot results from DAE or VAE

Generate plots of the test results by
```
python dae_plot.py
python vae_plot.py
```

### Slurm scripts

Alternatively, edit and run the slurm script
```
sbatch run_dae.s
sbatch run_vae.s
```

## Compare models

Generate plot comparing the multiple methods above,
run in the ``scripts`` subdirectory
```
python plot_metrics.py
```

