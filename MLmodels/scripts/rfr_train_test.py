"""
Train and Test ML Model for the SJdF Data. Outputs predictions and true solution as .txt files
Authors: Chris Liu
"""

import numpy as np
import tsunami_regress as tsr
import os
import importlib
import time

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import explained_variance_score

from tsfresh import extract_features, select_features
from tsfresh.feature_selection import relevance
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings

if __name__ == "__main__":

    np.random.seed(10000)
    
    # Timing the code
    time_s = time.perf_counter()
    
    # Load Data
    runs = range(0,1300)
    gauges = [702,901,911]
    modelname = 'RFR'

    ddir = r'../data/_sjdf/SJdF_processed_gauge_data' # location of interpolated data
    outdir = r'_output' # Directory containing predictions and true data.
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    indexdir = r'../data/_sjdf' # Directory containing indices.
    path = r'../data/_sjdf' #path containing all the indices
    
    # Pathes and names of the non-KL data. L1, B-Whole, S-A Whole, and wang1700
    vpaths = [r'../data/_sjdf/L1',
              r'../data/_sjdf/B-Whole',
              r'../data/_sjdf/S-A-Whole',
              r'../data/_sjdf/wang1700']
    
    nonkl = ['L1','B-Whole','S-A-Whole','wang1700']
    
    # Load time series
    eta, t= tsr.load_data(ddir, runs, gauges)
    
    # Load txt file containing the run numbers used
    runnos = np.loadtxt(os.path.join(path,'sjdf_runno.txt')).astype(int)
    
    # Import the run numbers and indices for test/train sets
    runnos_train = np.loadtxt(os.path.join(path,'sjdf_train_runno.txt')).astype(int)
    runnos_test = np.loadtxt(os.path.join(path,'sjdf_test_runno.txt')).astype(int)
    index_train = np.loadtxt(os.path.join(path,'sjdf_train_index.txt')).astype(int)
    index_test = np.loadtxt(os.path.join(path,'sjdf_test_index.txt')).astype(int)
    
    tsteps = [181, 361] # 30 minutes
    threshold = 0.1
    
    sc = False # Scale features
    
    # Stack the time series into a dictionary for featurization in tsfresh
    eta_g702 = []
    run_id = []
    runs_used = []
    times = []
    tstart = []

    for i in range(len(tsteps)):
        eta_tmp, run_id_tmp, runs_used_tmp, times_tmp, tstart_tmp =\
           tsr.stack_series(eta, t, 702, runnos, 901, 0.5, threshold, tsteps[i])

        eta_g702.append(eta_tmp)
        run_id.append(run_id_tmp)
        runs_used.append(runs_used_tmp)
        times.append(times_tmp)
        tstart.append(tstart_tmp)
    
    # Featurize using tsfresh
    feat702 = []

    for i in range(len(eta_g702)):
        dict = {'id':run_id[i], 'time':times[i], 'eta': eta_g702[i]}

        feat_temp = extract_features(pd.DataFrame(dict), column_id='id', column_sort='time',\
                            default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)

        # drop constant features
        feat702.append(feat_temp.loc[:, feat_temp.apply(pd.Series.nunique) != 1]) 
    
    # Extract feature settings for model prediction
    params_30min = settings.from_columns(feat702[0])
    params_60min = settings.from_columns(feat702[1])
    
    # Create targets for train/test
    g901max = tsr.max_eta(eta,901,runnos)
    g911max = tsr.max_eta(eta,911,runnos)
    
    # Specify the model
    rmodel = RandomForestRegressor(n_estimators=100)

    ## Gauge 901
    pred_901, trp_901, target_901, evs_901, scalers_901, models_901 = tsr.train_test(feat702, g901max, runnos,\
                                                                                     index_train, index_test,\
                                                                                    sc,'r', rmodel,True)
    ## Gauge 911
    pred_911, trp_911, target_911, evs_911, scalers_911, models_911 = tsr.train_test(feat702, g911max, runnos,\
                                                                                     index_train, index_test,\
                                                                                  sc, 'r', rmodel, True)
    
    # End timer
    time_e = time.perf_counter()
    
    # Verify models on non-KL 
    
    nonkl_true = np.zeros((len(vpaths),2))
    nonkl_pred = np.zeros((2,len(vpaths),2))
    
    for j, vpath in enumerate(vpaths):
        path702 = os.path.join(vpath,'gauge00702.txt')
        
        eta_v, t_v = tsr.load_verif_test(path702, 10)
        
        # set a value of 0 to the first data point since it is NaN due to interp. 
        # Poor results when I don't capture that initial drop
        eta_v[0]=0 
        
        feat_702_v = [tsr.feat_verif_test(eta_v, t_v, 181, params_30min), 
                 tsr.feat_verif_test(eta_v, t_v, 361, params_60min)]
        
        pred_901_v = tsr.predict(scalers_901,models_901,feat_702_v)
        pred_911_v = tsr.predict(scalers_911,models_911,feat_702_v)
        
        # True values
        nonkl_true[j,0] = tsr.verif_max_eta(vpath, 901)
        nonkl_true[j,1] = tsr.verif_max_eta(vpath, 911)
        
        # Predicted values values
        nonkl_pred[0,j,:] = pred_901_v 
        nonkl_pred[1,j,:] = pred_911_v
        
        
    # Save results for SJDF
    sizetst = len(index_test)
    sizetr =  len(index_train)

    # save the true results
    g901max_test = g901max[index_test].reshape((sizetst,1))
    g911max_test = g911max[index_test].reshape((sizetst,1))
    np.savetxt(os.path.join(outdir, 'etamax_obs_10s.txt'),\
               np.hstack((g901max_test, g911max_test)))
    np.savetxt(os.path.join(outdir, 'etamax_obs_10s_all.txt'),\
               np.hstack((g901max.reshape((sizetst+sizetr,1)), g911max.reshape((sizetst+sizetr,1)))))
    
    # Save model predictions
    for i in range(len(tsteps)): 
        winsize = int((tsteps[i]-1)*10/60)
        etamax_pred_tst = np.hstack((pred_901[i].reshape((sizetst,1)),pred_911[i].reshape((sizetst,1))))
        etamax_pred_tr = np.hstack((trp_901[i].reshape((sizetr,1)),trp_911[i].reshape((sizetr,1))))
        
        np.savetxt(os.path.join(outdir, 'etamax_%s_predict_%sm.txt' %  (modelname, str(winsize))),  etamax_pred_tst)
        np.savetxt(os.path.join(outdir, 'etamax_%s_predict_tr_%sm.txt' %  (modelname, str(winsize))),  etamax_pred_tr)
        
        # Save results for non-KL
        
        for ii in range(len(nonkl)):
            nonkltest = nonkl[ii]

            np.savetxt(os.path.join(outdir,'etamax_%s_predict_%s_%sm.txt') %\
                       (modelname, nonkltest, str(winsize)), nonkl_pred[:,ii,i].reshape((1,-1)))
            np.savetxt(os.path.join(outdir,'etamax_obs_%s.txt' % nonkltest), nonkl_true[ii].reshape((1,-1)))

    # Save elapsed time
    tot_time = time_e-time_s
    timelog = open("%s_timelog.txt" % modelname,"a") 
    timelog.write(str(tot_time) + '\n')
    timelog.close()
