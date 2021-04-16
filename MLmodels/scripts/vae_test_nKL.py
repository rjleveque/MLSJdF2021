#!/usr/bin/env python3
r"""
Predict L1 test scenario: a special scenario generated from a different
earthquake generation method

"""

import os
import numpy as np
import vae

if __name__ == "__main__":

    # load model, use cpu device (gpu performance is not needed for prediction)
    AE = vae.VarAutoEncoder() 
    AE.load_model(model_name='vae_sjdf', device='cpu')

    # predict using model trained up to 500 epochs
    epoch = 750

    # interpolate L1 gauge results 
    gauges = AE.gauges
    ngauges = AE.ngauges
    npts = 256

    nKL_name_list = ['L1', 'B-Whole', 'S-A-Whole', 'wang1700']

    for nKL_name in nKL_name_list:

        data_path = os.path.join('../data/_sjdf', nKL_name)
        gauge_input = np.zeros((1, ngauges, npts))
        etamax_obs = np.zeros((1,ngauges))
        etamax_pred = np.zeros((1,ngauges))
        
        for k, gauge in enumerate(gauges):
            fname = 'gauge{:05d}.txt'.format(gauge)
            load_fname = os.path.join(data_path, fname)
            raw_gauge = np.loadtxt(load_fname, skiprows=3)
            t = raw_gauge[:, 1]
            eta = raw_gauge[:, 5]
            etamax_obs[0, k] = eta.max()
            if k == 0:
                # set prediction time-window
                t_init = np.min(t[np.abs(eta) > 0.1])
                t_final = t_init + 5*3600.0
                t_unif = np.linspace(t_init, t_final, npts)
            
            # interpolate to uniform grid on prediction window
            gauge_input[0, k, :] = np.interp(t_unif, t, eta)
            
        pred_all = AE.predict_input(gauge_input, epoch)
        pred_all = np.array(pred_all)

        # save predicted time-series
        save_fname = '_output/vae_sjdf_test_{:s}_e{:04d}.npy'.format(
                                                              nKL_name,epoch)
        np.save(save_fname, pred_all)

        # save interpolated time-series version of observations
        save_fname = '_output/vae_sjdf_obs_{:s}.npy'.format(nKL_name)
        np.save(save_fname, gauge_input)

        # save true etamax from raw data
        save_fname = '_output/etamax_obs_{:s}.txt'.format(nKL_name)
        np.savetxt(save_fname, etamax_obs)
        
        # save predicted etamax

        minutes = [60, 30]
        for i in range(2):
            fname = '_output/etamax_VAE_predict_{:s}_{:02d}m.txt'.format(
                                                         nKL_name, minutes[i])

            etamax_pred = pred_all.max(axis=-1).mean(axis=1).squeeze()[[i], :]
            np.savetxt(fname, etamax_pred)
