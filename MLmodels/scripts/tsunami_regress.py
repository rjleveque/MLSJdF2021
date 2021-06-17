"""
4/8/2021

Documentation updated, 6/16/2021

Module for tsunami regression.

@author: Chris Liu
"""

import numpy as np
import pandas as pd
import math
import os
from pylab import *
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import explained_variance_score, accuracy_score

from tsfresh import extract_features, select_features
from tsfresh.feature_selection import relevance
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings

"""
Loading data, data exploration
"""  
  
def load_data(outdir, rnums, gaugenos):
    """
    Loads data from specified runs and gauge numbers
    
    Parameters
    ----------
        outdir :
            outer directory containing runs and corresponding gauge data
        rnums :
            Run numbers to be loaded
        gaugenos :
            Gauge numbers to be loaded
        
    Returns
    ----------
        eta :
            Dictionary of eta data with the form: eta[(run_number, gauge_number)]
        t :  
            Dictionary of time intervals of the form: t[(run_number, gauge_number)]
    """
    eta = {}
    t = {}
    for rnum in rnums:
        rundir = os.path.join(outdir, 'run_%s' % str(rnum).zfill(6))
        for gaugeno in gaugenos:
            gfile = '%s/gauge%s.txt' % (rundir, str(gaugeno).zfill(5))
            g_data = np.genfromtxt(gfile, delimiter=',')

            # data in this case has a NaN value for eta @ t=0
            t[(rnum,gaugeno)] = g_data[1:,0]  # seconds
            eta[(rnum,gaugeno)] = g_data[1:,1]  # surface elevation in meters
    
    return eta, t

"""
Pre-processing, label creation
"""

def max_eta(data,gaugeno,runs):
    """
    Finds the max eta for a specific gauge and run(s)
    
    Parameters
    ----------
        data :
            Dictionary of timeseries data with the form: data[(run_number, gauge_number)]
        gaugeno :
            Integer gauge number
        runs :
            List or range of run numbers to find the max value for
    
    Returns
    ----------
        eta_max :
            npy array containing the maximum values.
    """
    eta_max = []
    for rnum in runs:
        eta_max.append(np.amax(data[(rnum,gaugeno)]))
    return np.array(eta_max)

def get_thresh(data,threshold):
    """
    Finds index of time series when threshold is met or exceeded
    
    Parameters
    ----------
        data :
            npy array of time series
        threshold :
            int/double that each entry in the time series is compared against 
        
    Returns
    ----------
        i :
            Integer index of time series when threshold is met or exceeded
                OR
            math.nan - Returns NaN if threshold is not met/exceeded
    """
    for i in range(len(data)):
        if np.abs(data[i]) >= threshold:
            return i
    return math.nan


def stack_series_raw(data, time, gaugeno, rnums, gnum_thresh, g_thresh, threshold, tsteps):
    """
    Stacks time series directly into an npy array with thresholding for input to ML model. 
    Use when feature extraction is not desired.
    
    Parameters
    ----------
        data :
            Dictionary of eta data with the form: data[(run_number, gauge_number)]
        time :
            Dictionary of time intervals of the form: time[(run_number, gauge_number)]
        gaugeno :
            Integer gauge number
        runs :
            Integer total number of runs
        gnum_thresh :
            Integer gauge number of target gauge used in prediction we are also thresholding
        g_thresh :
            Threshold eta of target gauge
        threshold :
            int/double that each entry in the time series is compared against for gaugeno
        tsteps :
            Number of time steps to extract after threshold is met
        
    Returns
    ----------
        g :
            npy array of eta time series with dimensions [#realizations x length of time series]
        runs_used :
            List containing the run numbers used (met the threshold). Not used for feature extraction
        times :
            npy array of  time scale/interval corresponding to an eta time series 
            with dimensions [#realizations x length of time series]
        tstart :
            The indices where each time series met or exceeded the threshold. Not used for feature extraction
    """
    runs_used = []
    g = []
    times = []
    tstart = []
    
    for rnum in rnums:
        g_data = data[(rnum,gaugeno)]
        t_data = time[(rnum,gaugeno)]
        w_ind = get_thresh(g_data,threshold) #returns NaN if threshold is not met/exceeded for input window
        g_ind = get_thresh(data[(rnum,gnum_thresh)],g_thresh) #returns NaN if threshold is not met/exceeded for target gauge
        
        if not math.isnan(w_ind) and not math.isnan(g_ind):
            #checking to see if there is enough data after threshold is met/exceeded
            if w_ind+tsteps < len(g_data):
                runs_used.append(rnum)
                tstart.append(w_ind)
                g.append(g_data[w_ind:w_ind+tsteps])
                times.append(t_data[w_ind:w_ind+tsteps])
                
    return np.asarray(g), runs_used, np.asarray(times), tstart

def stack_series(data, time, gaugeno, rnums, gnum_thresh, g_thresh, threshold, tsteps):
    
    """
    Stacks the time series used to construct the dataframe for feature extraction with thresholding.
    
    Parameters
    ----------
        data :
            Dictionary of eta data with the form: data[(run_number, gauge_number)]
        time :
            Dictionary of time intervals of the form: time[(run_number, gauge_number)]
        gaugeno :
            Integer gauge number
        runs :
            Integer total number of runs
        gnum_thresh :
            Integer gauge number of target gauge used in prediction we are also thresholding
        g_thresh :
            Threshold eta of target gauge
        threshold :
            int/double that each entry in the time series is compared against for gaugeno
        tsteps :
            Number of time steps to extract after threshold is met
        
    Returns
    ----------
        g :
            Stacked eta time series of the form [(Run#0000,Gauge#702),(Run#0001,Gauge#702), ..., (Run#1299,Gauge#702)]
        run_id :
            ID used to identify which run number the time series belongs to, ranges from 0 to 1299
        runs_used :
            List containing the run numbers used (met the threshold). Not used for feature extraction
        times :
            The time scale/interval corresponding to an eta time series
        tstart :
            The indices where each time series met or exceeded the threshold. Not used for feature extraction
    """
    runs_used = []
    g = []
    times = []
    run_id = []
    tstart = []
    
    for rnum in rnums:
        g_data = data[(rnum,gaugeno)]
        t_data = time[(rnum,gaugeno)]
        w_ind = get_thresh(g_data,threshold) #returns NaN if threshold is not met/exceeded for input window
        g_ind = get_thresh(data[(rnum,gnum_thresh)],g_thresh) #returns NaN if threshold is not met/exceeded for target gauge
        
        if not math.isnan(w_ind) and not math.isnan(g_ind): 
            runs_used.append(rnum)
            tstart.append(w_ind)
            
            #checking to see if there is enough data after threshold is met/exceeded
            if w_ind+tsteps < len(g_data):
                g.extend(g_data[w_ind:w_ind+tsteps].tolist())
                times.extend(t_data[w_ind:w_ind+tsteps].tolist())
                run_id.extend((np.ones(tsteps)*rnum).tolist())
            else:
                g.extend(g_data[w_ind:].tolist())
                times.extend(t_data[w_ind:].tolist())
                run_id.extend((np.ones(len(g_data)-w_ind)*rnum).tolist())

    return g, run_id, runs_used, times, tstart

def stack_series_all(data, time, gaugeno, runs):
    """
    Stacks the time series used to construct the dataframe for feature extraction without thresholding.
    
    Parameters
    ----------
        data :
            Dictionary of eta data with the form: data[(run_number, gauge_number)]
        time :
            Dictionary of time intervals of the form: time[(run_number, gauge_number)]
        gaugeno :
            Integer gauge number
        runs :
            Integer total number of runs
        
    Returns
    ----------
        g :
            Stacked eta time series of the form [(Run#0000,Gauge#702),(Run#0001,Gauge#702), ..., (Run#1299,Gauge#702)]
        run_id :
            ID used to identify which run number the time series belongs to, ranges from 0 to 1299
        times :
            The time scale/interval corresponding to an eta time series
    """
    
    rnums = range(0,runs)
    
    g = []
    times = []
    run_id = []
    
    for rnum in rnums:
        g_data = data[(rnum,gaugeno)]
        t_data = time[(rnum,gaugeno)]
        
        g.extend(g_data.tolist())
        times.extend(t_data.tolist())
        run_id.extend((np.ones(len(g_data))*rnum).tolist())

    return g, run_id, times

def max_eta_all(data,gaugeno,runs):
    """
    Finds the max eta for all runs and a specific gauge 
    
    Parameters
    ----------
        data :
            Dictionary of timeseries data with the form: data[(run_number, gauge_number)]
        gaugeno :
            Integer gauge number
        runs :
            Number of runs
    
    Returns
    ----------
        eta_max :
            npy array containing the maximum values.
    """
    eta_max = []
    for rnum in range(runs):
        eta_max.append(np.amax(data[(rnum,gaugeno)]))
    return np.array(eta_max)

def classify_labels(maxeta,cat):
    """
    Creates labels for classifcation from the max eta values by binning each
    realization
    
    Parameters
    ----------
        maxeta :
            npy array containig max eta values
        cat :
            bin edges
    Returns
    ----------
        npy array of classification labels
    """
    labels = []
    for gmax in maxeta:
        if gmax < cat[0]:
            labels.append('A')
        elif gmax < cat[1]:
            labels.append('B')
        elif gmax < cat[2]:
            labels.append('C')
        else:
            labels.append('D')
    return np.array(labels)

def train_test_split(data, target, train_ind, test_ind):
    """
    Splits ML input and targets into training and test sets
    from the specified indices
    
    Parameters
    ----------
        data :
            npy array of the input time series
        target :
            npy array of ML targets
        train_ind :
            list of training indices
        test_ind :
            list of test indicies
    Returns
    ----------
        train :
            npy array of time series training set
        test  :
            npy array of time series test set
        train_target :
            npy array of training targets
        test_target :
            npy array of test targets
    """
    train = data[train_ind,:]
    train_target = target[train_ind]
    test = data[test_ind,:]
    test_target = target[test_ind]
    return train, test, train_target, test_target

"""
Build/Train/Test Model
"""
def train_test(data, target, train_ind, test_ind, scale, c_or_r, model, *returns):
    """
    Trains and tests the model using non-linear SVR/SVC. 
    
    Parameters
    ----------
        data :
            Input data to the model, can be raw npy array or feature dataframes
        target :
            Array of target values for regression
        train_ind :
            List of indices for training set
        test_ind :
            List of indices for testing set
        scale :
            Boolean value to denote whether features should be scaled or not to unit variance and 0 mean.
        c_or_r :
            Value for specifying classification or regression
        model :
            sklearn model to be used
        *returns :
            Boolean value to determine whether scalers and models need to be returned.
        
    Returns
    ----------
        pred :
            List of arrays of predictions from testing the model after it is trained
        tr_pred:
            List of arrays of predictions of the training set after model training
        target :
            List of arrays of targets that correspond to runs in the test set
        acc :
            List of accuracy scores for each dataset.
        scalers :
            List of standard scalers used (Optional)
        models :
            List of trained models (Optional)
    """
    data_tmp = data
    pred = []
    tr_pred = []
    targets = []
    acc = []
    
    if returns:
        scalers = [] # empty is scale = false
        models = []

    for i in range(len(data)):
        
        # Check file format
        if isinstance(data_tmp[i],pd.DataFrame):
            train_set, test_set, train_target, test_target, = \
                train_test_split(data_tmp[i].to_numpy(), target, train_ind, test_ind)
        else:
            train_set, test_set, train_target, test_target, = \
                train_test_split(data_tmp[i], target, train_ind, test_ind)            
        
        if scale:
            scaler = StandardScaler()
            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set)
        
        model_tmp = clone(model)
        
        model_tmp.fit(train_set, train_target, sample_weight=None)
        
        pred.append(model_tmp.predict(test_set))
        tr_pred.append(model_tmp.predict(train_set))
        
        if c_or_r == 'r':
            acc.append(explained_variance_score(test_target,pred[i]))
        else:
            acc.append(accuracy_score(test_target,pred[i]))
            
        targets.append(test_target)
        
        if returns:
            models.append(model_tmp)
            if scale:
                scalers.append(scaler)
            
    if returns:
        return pred, tr_pred, targets, acc, scalers, models
    else:
        return pred, tr_pred, targets, acc

"""
Exploring/Plotting Model Results
"""
def find_inacc_runs(pred, target, runs, n):
    
    """
    Finds runs of the n largest absolute difference between predicted and 
    actual and outputs a pandas dataframe. 

    Parameters
    ----------
       pred :
           Prediction from regression model
       target :
           Actual max eta value
       runs :
           run numbers used for the testing set
       n :
           number of largest absolute differences returned

    Returns
    ----------
        pd.Dataframe :
            pandas datafram containing the run numbers, predicted and actual eta, 
            and absolute difference of predicted and actual.
    """
    
    difference = np.abs(pred-target)
    ind = difference.argsort()[-n:][::-1]
    
    run_dict = {'Run Number' : np.array(runs)[ind], 'Predicted' : pred[ind], 'Actual' : target[ind], 
                    'Abs Diff': difference[ind]}
    
    return pd.DataFrame(data = run_dict)

def find_run(pred, target, runs, run_num):
    """
    Prints predicted and actual max eta for a specified run. 

    Parameters
    ----------
       pred :
           Prediction from regression model
       target :
           Actual max eta value
       runs :
           run numbers used for the testing set
       run_num :
           specified run number

    """

    ind = np.where(runs == run_num)
    
    if len(ind[0]) == 1:
        print("Run Number:" + str(run_num) + ", Predicted:" + str(pred[ind]) + ", Actual:" + str(target[ind]))
    else:
        print("Run not found.")

def plot_test_all(target,pred,line,zoomlim, labels):
    """
    Plots the predicted versus actual value for both datasets along
    with a reference line of slope 1.
    
    Parameters
    ----------
        target :
            Target values
        pred :
            Predicted values
        line :
            Endpoint of reference line
        zoomlim :
            x and y limit for the zoomed in subplot
        labels :
            labels for the legend
    """
    #legend
    
    fig = figure(figsize=(13,7))
    
    ax = fig.add_subplot(1,2,1)
    
    plot(target[0],pred[0],color='tab:orange',marker='.',linestyle='None')  
    plot(target[1],pred[1],color='tab:blue',marker='.',linestyle='None')
    
    plot([0,line],[0,line],'k--')
    
    ax.legend(labels)
    
    xlabel('Max eta (Actual)')
    ylabel('Max eta (Predicted)')
    
    xlim(0,line)
    ylim(0,line)
    ax.set_aspect('equal', adjustable='box')
    
    grid(True)
    
    ax2 = fig.add_subplot(1,2,2)
    
    plot(target[0],pred[0],color='tab:orange',marker='.',linestyle='None')  
    plot(target[1],pred[1],color='tab:blue',marker='.',linestyle='None')
    
    plot([0,zoomlim],[0,zoomlim],'k--')
    
    xlabel('Max eta (Actual)')
    ylabel('Max eta (Predicted)')
    
    xlim(0,zoomlim)
    ylim(0,zoomlim)
    ax2.set_aspect('equal', adjustable='box')
    
    grid(True)

def plot_test(target,pred,line,zoomlim,setnum):
    """
    Plots the predicted versus actual value for a single dataset along with a reference line of slope 1.
    
    Parameters
    ----------
        target :
            Target values
        pred :
            Predicted values
        line :
            Endpoint of reference line
        zoomlim :
            x and y limit for the zoomed in subplot
    """
    
    fig = figure(figsize=(13,7))
    
    ax = fig.add_subplot(1,2,1)
    
    plt.plot(target[setnum],pred[setnum],'.r')  
    
    plot([0,line],[0,line],'k--')
    
    xlabel('Max eta (Actual)')
    ylabel('Max eta (Predicted)')
    
    plt.xlim(0,line)
    plt.ylim(0,line)
    ax.set_aspect('equal', adjustable='box')
    
    grid(True)
    
    ax2 = fig.add_subplot(1,2,2)
     
    plt.plot(target[setnum],pred[setnum],'.r')
    
    plot([0,zoomlim],[0,zoomlim],'k--')
    
    xlabel('Max eta (Actual)')
    ylabel('Max eta (Predicted)')
    
    plt.xlim(0,zoomlim)
    plt.ylim(0,zoomlim)
    ax2.set_aspect('equal', adjustable='box')
    
    grid(True)
    
def plot_run(data, time, starttime, runs, rnum, gaugeno, inputg, winnum, winsize, grid, *pred):
    
    """
   Plots gauge 702 and a specified 9XX gauge for a specified run number (Temporarily commented out the horizontal lines)

   Parameters
   ----------
       data :
           Dictionary of eta data with the form: data[(run_number, gauge_number)]
       time :
           Dictionary of time intervals of the form: time[(run_number, gauge_number)]
       starttime :
           The indices where each time series met or exceeded the threshold.
       runs :
           List containing the run numbers used (met the threshold).
       rnum :
           Run number of interest
       gaugeno :
           Gauge number of interest
       inputg :
           Input gauge number
       winnum :
           Integer index of data set of interest
       winsize :
           Number of time steps to extract after threshold is met for the given dataset
       *pred :
           optional argument for printing predicted value on title of plot
    """
    
    fig, (ax1, ax2) = plt.subplots(2,sharex=True, sharey=False,figsize=(12,10))
    
    title = 'Run # %s' % rnum
    
    if pred:
        title =title + ', Predicted: %sm' % np.around(pred[0],2)
    
    ax1.set_title(title, fontsize=20,)
    
    fig.add_subplot(111, frameon=False) # used for centering the y-axis label
    
    ax1.plot(time[(rnum,inputg)]/60, data[(rnum,inputg)], label="Gauge # %s" % str(inputg), color='blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    
    #Plot window of data used and reference line for threshold. 
    start = starttime[winnum][runs.index(rnum)]*grid
    end = (start + (winsize[winnum]-1)*grid)
    ax1.axvline(start/60, color ='red', ls='--', lw=1, alpha = 0.8)
    ax1.axvline(end/60, color ='red', ls='--', lw=1, alpha = 0.8)

    ax2.plot(time[(rnum,gaugeno)]/60, data[(rnum,gaugeno)], label='Gauge # %s' % str(gaugeno), color='blue')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) # used for centering the y-axis label
    plt.xlabel('Minutes after quake', fontsize=16)
    plt.ylabel('Surface elevation (meters)', fontsize=16)
    
def save_run_plots(data, time, starttime, runs, rnums, pred, gaugeno, winnum, winsize):
    """
    Saves plots of eta for specified runs of gauge 702 and a specified gauge 9XX as a png.

    Parameters
    ----------
       data :
           Dictionary of eta data with the form: data[(run_number, gauge_number)]
       time :
           Dictionary of time intervals of the form: time[(run_number, gauge_number)]
       starttime :
           The indices where each time series met or exceeded the threshold.
       runs :
           List containing the run numgbers used (met the threshold).
       rnums :
           Run numbers of interest
       pred :
           Predicted values for runs of interest
       winnum :
           Integer index of data set of interest
       winsize :
           Number of time steps to extract after threshold is met for the given dataset
       gaugeno :
           Gauge number of interest
    """
    for i in range(len(rnums)):
        plot_run(data, time, starttime, runs, rnums[i], gaugeno, winnum, winsize, pred[i])
        plt.savefig("r%s_g%s.png" % (rnums[i],gaugeno))

"""
Predict
"""
def predict(scalers,models,feats):
    """
    Scale input features and return model prediction.
    
    Parameters
    ----------
        scalers :
           list of sklearn StandardScalers fit during the training process
        models :
            list of sklearn models
        feats :
            pd dataframe of input features
    Returns
    ----------
        pred :
            model prediction
    """
    scalers_f = scalers
    models_f = models
    
    pred = []

    for i in range(len(feats)):
        model_temp = models_f[i]
        ft_temp = feats[i]
        
        if len(scalers_f) != 0:
            scale_temp = scalers_f[i]
            
            if isinstance(ft_temp,pd.DataFrame):
                ft_temp = scale_temp.transform(ft_temp[ft_temp.columns])
            else:
                ft_temp = scale_temp.transform(ft_temp.reshape(1, -1))

        pred.append(model_temp.predict(ft_temp)[0])
    
    return pred

"""
Verifying Models
"""
def load_verif_test(file, interp):
    """
    Loads the additional non-KL data for verification
    
    Parameters
    ----------
        file : str
           filepath for time series
        interp : int
            Time increment to interpolate to
    Returns
    ----------
        eta_unif : 
           array-like of interpolated wave amplitude
        tt : 
           npy array of time steps corresponding to eta
    """
    gdata = np.loadtxt(file)
    
    t_tmp   = gdata[:,1] 
    eta_tmp = gdata[:,5]
    
    tt = np.arange(0., t_tmp[-1], interp)
    
    gaugefcn = interp1d(t_tmp , eta_tmp, kind='linear', bounds_error=False)
    eta_unif = gaugefcn(tt)
    
    return eta_unif, tt

def verif_max_eta(outdir, gaugeno):
    """
    Returns the max eta of time series
    
    Parameters
    ----------
        outdir :
           filepath of directory containing file
        gaugeno : int
            gauge number of interest
    Returns
    ----------
        max eta of time series
    """
    gdata = np.loadtxt(os.path.join(outdir, 'gauge%s.txt' % str(gaugeno).zfill(5)))
    
    return np.amax(gdata[:,5])

def feat_verif_test(eta_u, tt_u, tstep, params):
    """
    Featurizes interpolated time series for non-KL data using tsfresh
    
    Parameters
    ----------
        eta_u :
           interpolated wave eta
        tt_u : 
            associated time steps
        tstep :
            Input time window
        params :
            tsfresh feature parameters
    Returns
    ----------
        feat_tmp :
            pd dataframe of features
    """
    dict_tmp = {'id':np.ones(len(eta_u[0:tstep])), 'time':tt_u[0:tstep], 'eta': eta_u[0:tstep]}
    feat_tmp = extract_features(pd.DataFrame(dict_tmp), column_id='id', column_sort='time', 
                                   kind_to_fc_parameters=params) 
    
    return feat_tmp

"""
Old Functions, Documentation is likely not up to date
"""

def train_test_split_old(data, target, runs_used, test_size, seed):
    """
    Splits data and target into training and testing sets for a given random seed. Keeps track of which runs are
    put into training and testing sets.
    
    Parameters
    ----------
        data :
            Dataframe of data samples
        target :
            np array of targets
        runs_used :
            np array of run numbers used
        test_size :
            test size as a fraction of the total samples (between 0 and 1)
        seed :
            seed used for random number generator
    Output
        train :
            Dataframe of training data
        test :
            Dataframe of testing data
        train_target :
            np array of training targets
        test_target :
            np array of testing targets
        train_runs :
            np array of run numbers in training data
        test_runs :
            np array of run numbers in testing data
    
    """
    np.random.seed(seed)
    
    total = len(target)
    
    tt_size = np.round(total*test_size)
    tr_size = int(total - tt_size)
    
    perm = np.random.permutation(total)
    
    train = data[perm[:tr_size],:]
    train_target = target[perm[0:tr_size]]
    train_runs = runs_used[perm[0:tr_size]]
    
    test = data[perm[tr_size:],:]
    test_target = target[perm[tr_size:]]
    test_runs = runs_used[perm[tr_size:]]
    
    return train, test, train_target, test_target, train_runs, test_runs

def train_test_old(feat, target, runs_used, testsize, rseed, scale, c_or_r, model, *returns):
    """
    Trains and tests the model using non-linear SVR/SVC. 
    
    Parameters
    ----------
        feat :
            List of feature dataframes
        target :
            Array of target values for regression
        runs_used :
            Array of run numbers in the model
        scale :
            Boolean value to denote whether features should be scaled or not to unit variance and 0 mean.
        c_or_r :
            Value for specifying classification or regression
        *returns :
            Boolean value to determine whether scalers and models need to be returned.
        
    Returns
    ----------
        pred :
            List of arrays of predictions from testing the model after it is trained
        target :
            List of arrays of targets that correspond to runs in the test set
        runs :
            List arrays of runs used in testing.
        acc :
            List of accuracy scores for each dataset.
        scalers :
            List of standard scalers used (Optional)
        models :
            List of trained models (Optional)
    """
    
    pred = []
    targets = []
    tr_pred = []
    runs = []
    acc = []
    
    if returns:
        scalers = []
        models = []
    
    for i in range(len(feat)):
        train_set, test_set, train_target, test_target, train_runs, test_runs = \
            train_test_split_old(feat[i].to_numpy(), target, np.asarray(runs_used[i]),testsize,rseed)
        
        if scale:
            scaler = StandardScaler()
            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set)
        
        model_tmp = clone(model)
        
        model_tmp.fit(train_set, train_target, sample_weight=None)
        
        pred.append(model_tmp.predict(test_set))
        tr_pred.append(model_tmp.predict(train_set))
        
        if c_or_r == 'r':
            acc.append(explained_variance_score(test_target,pred[i]))
        else:
            acc.append(accuracy_score(test_target,pred[i]))
            
        runs.append(test_runs)
        targets.append(test_target)
        
        if returns:
            models.append(model_tmp)
            if scale:
                scalers.append(scaler)
            
    if returns:
        return pred, tr_pred, targets, runs, acc, scalers, models
    else:
        return pred, tr_pred, targets, runs, acc