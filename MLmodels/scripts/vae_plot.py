#!/usr/bin/env python3
r"""
Plot autoencoder prediction results

"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from aeplot import *

if __name__ == "__main__":

    # load dataset prediction
    model_name = 'vae_sjdf'
    obs_win_list = [ 51,  26]
    epochs_list =  [750, 750]
    nepochs = 1250
    vaeruns = 100

    if not os.path.exists('_plots'):
        os.mkdir('_plots')

    npts = 256
    t_unif = np.linspace(0.0, 5.0*60, npts)
    gauges = [702, 901, 911]

    nKL_name_list = ['L1', 'B-Whole', 'S-A-Whole', 'wang1700']

    fmt = 'png'
    dpi = 300

    # load obs data, pred results
    pred_train, obs_train, pred_test, obs_test, train_runno, test_runno \
                  = load_obspred(model_name, obs_win_list, epochs_list)
    
    ##
    ## Prediction vs. Observation plot
    ##

    for k, T in enumerate(obs_win_list):
        # load prediction
        epoch = epochs_list[k]
        
        pred_train_obs_win = pred_train[k]
        pred_test_obs_win = pred_test[k]

        # create plot
        fig_list = plot_obsvpred(pred_train_obs_win, obs_train, 
                                 pred_test_obs_win,  obs_test)

        for i, item in enumerate(fig_list):

            gauge = gauges[i]
            fig, ax, legends = item


            # add nKL results
            nKL_legends = legends
            markers = ['h','p','*','^']

            for ii, nKL_name in enumerate(nKL_name_list):
                minutes = int(t_unif[T]*60)
                fname = '_output/vae_sjdf_test_{:s}_e{:04d}.npy'.format(
                                                         nKL_name, epoch)
                pred_nKL = np.load(fname)

                etamax_pred_nKL = pred_nKL.max(axis=-1).mean(axis=1)[k, ...]\
                                    .squeeze()
                etamax_2std_nKL = 2.0*pred_nKL.max(axis=-1).std(axis=1)[k, ...]\
                                    .squeeze()

                fname = '_output/etamax_obs_{:s}.txt'.format(nKL_name)
                etamax_obs_nKL = np.loadtxt(fname)

                line0, = ax.plot(etamax_obs_nKL[i], 
                                 etamax_pred_nKL[i], 
                                 linewidth=0,
                                 marker=markers[ii],
                                 markersize=5,
                                 color='mediumblue',
                                 zorder=10)

                ax.errorbar(etamax_obs_nKL[i],
                            etamax_pred_nKL[i],
                            yerr=etamax_2std_nKL[i] , 
                            fmt = '.',
                            color='mediumblue',
                            elinewidth=0.8,
                            alpha = 0.3, 
                            capsize=0,
                            markersize=2)

                nKL_legends[0].append(line0)
                nKL_legends[1].append(nKL_name)

            ax.legend( nKL_legends[0], nKL_legends[1])

            # set title / save figures
            title = "gauge {:3d}, observation window {:d} mins"\
                    .format(gauge, int(t_unif[T]))

            ax.set_title(title, fontsize=10)

            fname = r'{:s}_predvobs_g{:03d}_t{:03d}_e{:04d}.{:s}'\
                    .format('vae', gauge, T, epoch, fmt)
            save_fname = os.path.join('_plots', fname)
        
            sys.stdout.write('\nsaving fig to {:s}'.format(save_fname))
            sys.stdout.flush()

            fig.tight_layout()

            sys.stdout.write('\nsaving fig to {:s}'.format(save_fname))
            fig.savefig(save_fname, dpi=300)

            # clean up
            plt.close(fig)     



    ##
    ## Plot training loss
    ##

    loss_all = np.zeros((2, vaeruns, nepochs+1))

    for k, T in enumerate(obs_win_list):
        for n in range(vaeruns):
            fname = '_output/vae_sjdf_train_loss_{:02d}_{:02d}.npy'\
                    .format(T, 1)
            loss_all[k, n, :] = np.load(fname)[:nepochs+1]

    # training loss plot
    s0 = 0.8
    fig, ax = plt.subplots(figsize=(s0*6,s0*4))

    colors = ['b', 'r']

    loss_mean = np.mean(loss_all, axis=1)
    loss_2std = 2.0*np.std(loss_all, axis=1)
    
    line_list = []
    label_list = []
    for k, T in enumerate(obs_win_list):
        line0, = ax.semilogy(loss_mean[k, 1:], 
                            color=colors[k])

        line_list.append(line0)
        
        t_pred = int(t_unif[T])
        label_list.append('{:s} ({:02d}mins)'\
                          .format('$\mathcal{L}$ ',t_pred))

    ax.set_ylabel('training loss')
    ax.set_xlabel('epochs')
    ax.set_xlim([0, nepochs])
    line_list.reverse()
    label_list.reverse()
    ax.legend(line_list, label_list)
    fig.tight_layout()

    fname = 'vae_training_loss.{:s}'.format(fmt)
    save_fname = os.path.join('_plots', fname)
    sys.stdout.write('\nsaving fig to {:s}'.format(save_fname))
    fig.savefig(save_fname, dpi=dpi)
    
    plt.close(fig)

    ##
    ## Time-series plots for runs in the test set
    ##

    runno = 1127

    pred2_run = pred_test[:, :, test_runno == runno, :, :].squeeze()
    obs_run = obs_test[test_runno == runno, : , :].squeeze()

    fig, axes = plot_pred_timeseries(pred2_run, obs_run, obs_win_list,
                                     gauges=gauges,  
                                     scale=1.2, dpi=300)

    k2col = np.argsort(obs_win_list)
    for k, T in enumerate(obs_win_list):
        col = k2col[k]
        title = 'Realization #{:04d}, observation window {:d} mins'\
                .format(runno, int(t_unif[T]))
        axes[0, col].set_title(title)
    
    fig.tight_layout()
    fname = 'vae_timeseries_test_r{:04d}.{:s}'.format(runno, fmt)
    fname = os.path.join('_plots', fname)
    fig.savefig(fname, dpi=dpi)

    ##
    ## Time-series plots for non-KL runs 
    ##

    for nKL_name in nKL_name_list:
        
        fname = 'vae_sjdf_obs_{:s}.npy'.format(nKL_name)
        load_fname = os.path.join('_output', fname)
        obs_nKL = np.load(load_fname).squeeze()

        pred2_nKL = []
        for k, T in enumerate(obs_win_list):
        
            epoch = epochs_list[k] 

            fname = 'vae_sjdf_test_{:s}_e{:04d}.npy'.format(nKL_name, epoch)
            load_fname = os.path.join('_output', fname)
            pred_nKL = np.load(load_fname) 
            pred2_nKL.append(pred_nKL[k, ...])    # TODO fixed indexing

        pred2_nKL = np.array(pred2_nKL).squeeze()

        fig, axes = plot_pred_timeseries(pred2_nKL, obs_nKL, obs_win_list,
                                         gauges=gauges,  
                                         scale=1.2, dpi=300)

        k2col = np.argsort(obs_win_list)
        for k, T in enumerate(obs_win_list):
            col = k2col[k]
            title = '{:s}, observation window {:d} mins'\
                    .format(nKL_name, int(t_unif[T]))
            axes[0, col].set_title(title)
        
        fig.tight_layout()
        fname = 'vae_timeseries_{:s}.png'.format(nKL_name)
        fname = os.path.join('_plots', fname)
        fig.savefig(fname, dpi=dpi)

    ##
    ## Plot ensemble prediction trajectories
    ##

    gaugei = 1
    runno = 1127
    k = 1
    T = obs_win_list[k]

    pred_run = pred_test[k, :, test_runno == runno, gaugei, :].squeeze().T
    obs_run = obs_test[test_runno == runno, gaugei , :].squeeze()

    fig, ax = plot_pred_ensemble(pred_run, obs_run, gaugei=gaugei)
    title = 'Realization #{:d}, observation window {:d} mins'\
            .format(runno, int(t_unif[T]))
    ax.set_title(title)

    fname = 'vae_timeseries_test_r{:04d}_ens.png'.format(runno)
    fname = os.path.join('_plots', fname)
    fig.savefig(fname, dpi=dpi)

    ##
    ## Time-series plots for runs in the test set
    ##

    for runno in test_runno:

        pred2_run = pred_test[:, :, test_runno == runno, :, :].squeeze()
        obs_run = obs_test[test_runno == runno, : , :].squeeze()

        fig, axes = plot_pred_timeseries(pred2_run, obs_run, obs_win_list,
                                         gauges=gauges,  
                                         scale=1.2, dpi=300)

        k2col = np.argsort(obs_win_list)
        for k, T in enumerate(obs_win_list):
            col = k2col[k]
            title = 'Realization #{:04d}, observation window {:d} mins'\
                    .format(runno, int(t_unif[T]))
            axes[0, col].set_title(title)
        
        fig.tight_layout()
        fname = 'vae_timeseries_test_r{:04d}.{:s}'.format(runno, fmt)
        fname = os.path.join('_plots', fname)
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)

    ##
    ## Time-series plots for non-KL runs 
    ##

    for nKL_name in nKL_name_list:
        
        fname = 'vae_sjdf_obs_{:s}.npy'.format(nKL_name)
        load_fname = os.path.join('_output', fname)
        obs_nKL = np.load(load_fname).squeeze()

        pred2_nKL = []
        for k, T in enumerate(obs_win_list):
        
            epoch = epochs_list[k] 

            fname = 'vae_sjdf_test_{:s}_e{:04d}.npy'.format(nKL_name, epoch)
            load_fname = os.path.join('_output', fname)
            pred_nKL = np.load(load_fname) 
            pred2_nKL.append(pred_nKL[k, ...])    # TODO fixed indexing

        pred2_nKL = np.array(pred2_nKL).squeeze()

        fig, axes = plot_pred_timeseries(pred2_nKL, obs_nKL, obs_win_list,
                                         gauges=gauges,  
                                         scale=1.2, dpi=300)

        k2col = np.argsort(obs_win_list)
        for k, T in enumerate(obs_win_list):
            col = k2col[k]
            title = '{:s}, observation window {:d} mins'\
                    .format(nKL_name, int(t_unif[T]))
            axes[0, col].set_title(title)
        
        fig.tight_layout()
        fname = 'vae_timeseries_{:s}.png'.format(nKL_name)
        fname = os.path.join('_plots', fname)
        fig.savefig(fname, dpi=dpi)

    ##
    ## Plot ensemble prediction trajectories
    ##

    gaugei = 1
    runno = 1127
    k = 1
    T = obs_win_list[k]

    pred_run = pred_test[k, :, test_runno == runno, gaugei, :].squeeze().T
    obs_run = obs_test[test_runno == runno, gaugei , :].squeeze()

    fig, ax = plot_pred_ensemble(pred_run, obs_run, gaugei=gaugei)
    title = 'Realization #{:d}, observation window {:d} mins'\
            .format(runno, int(t_unif[T]))
    ax.set_title(title)

    fname = 'vae_timeseries_test_r{:04d}_ens.png'.format(runno)
    fname = os.path.join('_plots', fname)
    fig.savefig(fname, dpi=dpi)
