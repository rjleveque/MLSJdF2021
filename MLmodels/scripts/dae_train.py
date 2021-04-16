#!/usr/bin/env python3
r"""
Train denoising autoencoder

"""

import os
import numpy as np
import dae
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    AE = dae.AutoEncoder(gauges=[702,901,911],\
                         model_name='dae_sjdf')
    # use GPU for training (e.g. on cluster)
    AE.device = 'cuda'

    # load data
    AE.load_data(batch_size=20, 
                 data_fname='../data/_sjdf/sjdf.npy') 

    # train ensemble
    AE.train_ensembles(nensemble=25, 
                       torch_loss_func=nn.L1Loss,
                       torch_optimizer=optim.Adam,
                       nepochs=500,
                       input_gauges=[702],
                       top=[51,26])  

