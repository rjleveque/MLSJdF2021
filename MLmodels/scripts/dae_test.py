#!/usr/bin/env python3
r"""
Predict test set

"""

import os
import numpy as np
import dae

if __name__ == "__main__":

    # load model, use cpu device (gpu performance not needed prediction)
    AE = dae.AutoEncoder() 
    AE.load_model(model_name='dae_sjdf', device='cpu')

    # predict using model trained up to 500 epochs
    #epoch = 100
    epochs_list = [200, 100]

    # predict etamax
    for epoch in epochs_list:
        AE.predict_dataset(epoch)

    # save obs / prediction to _output in ascii readable format
    for i, T in enumerate([26, 51]):

        epoch = epochs_list[i]
        fname = '_output/dae_sjdf_test_{:02d}_{:04d}.npy'.format(T, epoch)
        pred_all = np.load(fname)
        
        fname = '../data/_sjdf/sjdf.npy'
        data = np.load(fname)
        
        fname = '../data/_sjdf/sjdf_test_index.txt'
        test_index = np.loadtxt(fname).astype(np.int)
        obs_test = data[test_index, ...].max(axis=-1)

        ntest = len(test_index)

        pred_all = pred_all.max(axis=-1).mean(axis=0)
        pred_test = pred_all[-ntest:, :]

        vmax = np.max([pred_test.max(), obs_test.max()])

        if T == 26: minutes=30
        if T == 51: minutes=60

        fname = '_output/etamax_DAE_predict_{:02d}m.txt'.format(minutes)
        np.savetxt(fname, pred_test)

        fname = '_output/etamax_obs.txt'
        np.savetxt(fname, obs_test)

