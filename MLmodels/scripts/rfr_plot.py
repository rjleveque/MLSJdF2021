"""
Plot SJdF data
Authors: Donsub Rim
Modified by: Chris Liu
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

def plot_obsvpred(pred, train, obs, ind_tr, ind_tst, T, gauges, fmt='png'): 
    """
    Plot prediction vs. observation
    Input:
        pred - np array of predicted max eta values w/ test set
        train - np array of predicted max eta values w/ train set
        obs - np array of true observations
        ind_tr - Indices of the runs in the training set
        ind_tst -Indices of the runs in the test set
        T - int prediction input window length
        gauges - prediction gauges
    Output:
        fig_list - list a list of tuples containing 
            figure, axis, gauge number
    """
    fig_list = []

    for i, gauge in enumerate(gauges):
        
        fig, ax = plt.subplots()
        
        msg = '\r plotting / T={:4d}, gauge={:5d}'.format(T, gauge)
        sys.stdout.write(msg)
        sys.stdout.flush()

        obsi = obs[:, i]
        predi = pred[:, i]
        traini = train[:, i]
        obs_test = obsi[ind_tst]
        obs_train = obsi[ind_tr]

        vmax = max(predi.max(), obsi.max())

        fig,ax = plt.subplots(figsize=(4,4))
        ivmax = np.ceil(1.05*vmax)
        
        ax.plot([0.0, 1.05*ivmax],
                [0.0, 1.05*ivmax],
                "-k",
                linewidth=0.5)
        
        line0,= ax.plot( obs_train, 
                        traini,
                        "k.",
                        alpha=0.3,
                        markersize=2)
        
        line1, = ax.plot( obs_test, 
                         predi,
                         linewidth=0,
                         marker='D',
                         color='tab:orange',
                         markersize=1.5)


        legends = [[ line0,   line1],
                   ['train', 'test']]
        
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.grid(True, linestyle=':')
        ax.set_aspect("equal")

        ax.set_xlim([0.0, ivmax])
        ax.set_ylim([0.0, ivmax])

        if vmax > 10:
            ax.xaxis.set_ticks(np.arange(0.0, ivmax+1, 2))
            ax.yaxis.set_ticks(np.arange(0.0, ivmax+1, 2))
        else:
            ax.xaxis.set_ticks(np.arange(0.0, ivmax+1))
            ax.yaxis.set_ticks(np.arange(0.0, ivmax+1))
        fig_list.append((fig, ax, gauge, legends))

    return fig_list 

if __name__ == "__main__":
    
    outputdir = r'_output' # Directory containing predictions and true data.
    plotdir = r'_plots' # Directory containing predictions and true data.
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    indexdir = r'../data/_sjdf' # Directory containing indices.
    top = [30, 60] # time windows
    gauges = [901,911] # forecast gauges
    
    # nonKL tests
    nKL_name_list = ['L1', 'B-Whole', 'S-A-Whole', 'wang1700']

    mname = 'RFR'
    fmt = 'png'
    dpi = 300

    # load shuffled indices
    
    train_index = np.loadtxt(os.path.join(indexdir, 'sjdf_train_index.txt')).astype(int)
    test_index = np.loadtxt(os.path.join(indexdir, 'sjdf_test_index.txt')).astype(int)
    obs = np.loadtxt(os.path.join(outputdir, 'etamax_obs_10s_all.txt'))

    # Prediction vs. Observation plot
    for k, T in enumerate(top):
        # load prediction
        
        fname_tst = 'etamax_%s_predict_%sm.txt' % (mname,str(T))
        fname_tr = 'etamax_%s_predict_tr_%sm.txt' % (mname,str(T))
        load_fname_tr = os.path.join(outputdir, fname_tr)
        load_fname_tst = os.path.join(outputdir, fname_tst)
        pred = np.loadtxt(load_fname_tst)    
        train = np.loadtxt(load_fname_tr)
        
        fig_list = plot_obsvpred(pred, train, obs, train_index, test_index, T, gauges=gauges)
        for i,item in enumerate(fig_list):
            fig, ax, gauge, legends = item
            
            # add nKL results
            nKL_legends = legends
            markers = ['h','p','*','^']

            for ii, nKL_name in enumerate(nKL_name_list):
                
                # load non-kL
                etamax_obs_nKL = np.loadtxt(os.path.join(outputdir, 'etamax_obs_%s.txt' % nKL_name))
                pred_nKL = np.loadtxt(os.path.join(outputdir,\
                                                   'etamax_%s_predict_%s_%sm.txt' %\
                                                  (mname, nKL_name, T)))                
                
                line0, = ax.plot(etamax_obs_nKL[i], 
                                 pred_nKL[i], 
                                 linewidth=0,
                                 marker=markers[ii],
                                 markersize=5,
                                 color='mediumblue',
                                 zorder=10)

                nKL_legends[0].append(line0)
                nKL_legends[1].append(nKL_name)

            ax.legend( nKL_legends[0], nKL_legends[1])
            
            # save figures
            title = "gauge {:3d}, observation window {:d} mins"\
                    .format(gauge, T)
            ax.set_title(title, fontsize=10)

            fname = r'{:s}_predvobs_g{:03d}_t{:03d}.{:s}'\
                    .format(mname, gauge, T, fmt)
            
            save_fname = os.path.join(plotdir, fname) #windows file format
        
            sys.stdout.write('\rsaving fig to {:s}'.format(save_fname))
            sys.stdout.flush()
            fig.tight_layout()
            fig.savefig(save_fname, dpi=300)

            # clean up
            plt.close(fig)
