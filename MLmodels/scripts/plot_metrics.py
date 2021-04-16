
import numpy as np
import matplotlib.pyplot as plt


# sklearn models data
def read_obs_test(model, win, gauge):

    if (('RFR' in model) or ('SVR'  in model)):
        # standard test
        fname = '_output/etamax_obs_10s.txt'
        etamax_obs = np.loadtxt(fname)
        Ntest = etamax_obs.shape[0]

        if gauge == '901': gaugei = 0
        elif gauge == '911': gaugei = 1

        fname = '_output/etamax_{:s}_predict_{:s}.txt'\
                .format(model, win)
        etamax_test = np.loadtxt(fname)

        etamax_obs  = etamax_obs[:, gaugei]
        etamax_test = etamax_test[:, gaugei]

    elif (('DAE' in model) or ('VAE' in model)):
        # dae test
        fname = '_output/etamax_obs_10s.txt'
        etamax_obs = np.loadtxt(fname)
        Ntest = etamax_obs.shape[0]

        if gauge == '901': gaugei = 0
        elif gauge == '911': gaugei = 1

        fname = '_output/etamax_{:s}_predict_{:s}.txt'.format(model, win)

        etamax_test = np.loadtxt(fname)

        etamax_obs  = etamax_obs[:, gaugei]
        etamax_test = etamax_test[:, gaugei+1]
    else:
        print('the specified model output is not available.')
        raise


    return etamax_obs, etamax_test

from sklearn.metrics import explained_variance_score

model_list = ['SVRs', 'SVRs_raw', 'RFR', 'DAE', 'VAE']
gauge_list = ['901', '911']
win_list = ['30m', '60m']

vals_dict = {}

for win in win_list:
    for model in model_list:
        for gauge in gauge_list:
            etamax_obs, etamax_test = read_obs_test(model, win, gauge)

            Ntest = len(etamax_obs)
            l1error = np.linalg.norm(etamax_obs - etamax_test, ord=1)/Ntest
            evs = explained_variance_score(etamax_obs, etamax_test)
            vals_dict[ (model, win, gauge) ] = (l1error, evs)


# plot results
plt.close('all')

mrkr_dict = {}
mrkr_list = ['o', 'v', '^', 'o', 's', 'p']

color_dict = {}
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_list = [color_list[k] for k in [0,0,3,3,3,3,3]]

for model in model_list:
    mrkr_dict[model] = mrkr_list.pop()

sc = 1.05
#sc = 0.9
#fig, ax = plt.subplots(ncols=1, figsize=(sc*6,sc*5), sharex=True, sharey=True)
fig = plt.figure(figsize=(sc*8,sc*5))
ax = fig.add_axes([0.1, 0.125, 0.6, 0.8])

line_list_all = [[], []]
fillstyle = ['left', 'full']
markeredgecolor= [None, 'k']
for key, value in vals_dict.items():

    model, win, gauge = key

    if win == '30m' : i=0
    elif win == '60m' : i=1

    if gauge == '901' : j=0
    elif gauge == '911' : j=1

    #ax = axes[j]
    line_list = line_list_all[j]

    line0, = ax.plot(value[0], value[1], 
                    marker=mrkr_dict[model],
                    color=color_list[j + 3*i],
                    markeredgecolor=markeredgecolor[j],
                    markeredgewidth=1.5,
                    dashes=(1,0),
                    fillstyle=fillstyle[j],
                    markersize=11,
                    linewidth=0)
    
    ax.set_title('Regression metrics')

    if model == 'SVRs':
        model = 'SVM-tsf'
    label0 = '{:s} {:s} {:s}'\
             .format(gauge, win, model.replace('SVRs_raw','SVM-raw'))
    line_list.append( (line0, label0) )

leg_list = []
loc_list = [(1.05, 0.5), (1.05, 0.8)]
#for j in range(2):
#ax = axes[j]
line_list_all = line_list_all[0] + line_list_all[1]

leg0 = ax.legend([item[0] for item in line_list_all],
                 [item[1] for item in line_list_all],
                 bbox_to_anchor=(1.00, 1.02))
                 #prop={'size':9})
leg0.set_in_layout(False)
ax.set_xlabel('mean absolute error')

y0,y1 = ax.get_ylim()
w1 = y0
w0 = y1
ax.set_ylim([w0, w1])
ax.set_ylabel('explaned variance score')
fig.savefig('_plots/metrics.png', dpi=200)

fig.show()
