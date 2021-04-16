import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable

import copy

# 1D autoencoder
class Conv1DVAE(nn.Module):
    def __init__(self,ngauges,ninput, zdims):
        super(Conv1DVAE, self).__init__()

        self.ninput = ninput
        self.ngauges = ngauges
        self.zdims = zdims
       
        # encoder
        self.conv1 = nn.Conv1d(self.ninput, 64, 3, padding=1)  
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv1d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool1d(2, 2)
        self.ht = nn.LeakyReLU(negative_slope=0.5)

        self.fcnfor = nn.Linear(512, self.zdims*2)
        self.fcnback = nn.Linear(self.zdims, 512)
        # decoder
        self.t_conv1 = nn.ConvTranspose1d(512, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose1d(256, 256, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose1d(128, 128, 2, stride=2)
        self.t_conv6 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.t_conv7 = nn.ConvTranspose1d(64, 64, 2, stride=2)
        self.t_conv8 = nn.ConvTranspose1d(64, self.ngauges, 2, stride=2)


    # def encoder(self, x: Variable) -> (Variable, Variable):
    def encoder(self, x):
        x = self.ht(self.conv1(x))
        x = self.pool(x)
        x = self.ht(self.conv2(x))
        x = self.pool(x)
        x = self.ht(self.conv3(x))
        x = self.pool(x)
        x = self.ht(self.conv4(x))
        x = self.pool(x)
        x = self.ht(self.conv5(x))
        x = self.pool(x)
        x = self.ht(self.conv6(x))
        x = self.pool(x)
        x = self.ht(self.conv7(x))
        x = self.pool(x)
        x = self.ht(self.conv8(x))
        x = self.pool(x)

        x = x.view(x.shape[0], 512) #should be batch size
        x = self.fcnfor(x).view(-1, 2, self.zdims)
        mu = x[:, 0, :] # the first feature values as mean
        #should be batch x zdims (100 x 100)
        sig = x[:, 1, :]
        #get dimensions here, output mu and sigma
        return mu, sig
    # def reparameterize(self, mu: Variable, logvar: Variable) -> Variable: 
    def reparameterize(self, mu, logvar): 
        """The reparameterization is key to minimizing posteriors/priors

        For each training sample (we get some number batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [some dim, ZDIMS] mean matrix
        logvar : [som dim, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.
    """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        # if self.training:
        #     # multiply log variance with 0.5, then in-place exponent
        #     # yielding the standard deviation
        #     std = logvar.mul(0.5).exp_()  # type: Variable
        #     # - std.data is the [inner dims,ZDIMS] tensor that is wrapped by std
        #     # - so eps is [inner dims,ZDIMS] with all elements drawn from a mean 0
        #     #   and stddev 1 normal distribution that is number of inner feature samples
        #     #   of random ZDIMS-float vectors
        #     eps = Variable(std.data.new(std.size()).normal_())
        #     # - sample from a normal distribution with standard
        #     #   deviation = std and mean = mu by multiplying mean 0
        #     #   stddev 1 sample with desired std and mu, see
        #     #   https://stats.stackexchange.com/a/16338
        #     # - so we have #inner dims sets (the batch) of random ZDIMS-float
        #     #   vectors sampled from normal distribution with learned
        #     #   std and mu for the current input
        #     return eps.mul(std).add_(mu)
        # else:
        #     # During inference, we simply spit out the mean of the
        #     # learned distribution for the current input.  We could
        #     # use a random sample from the distribution, but mu of
        #     # course has the highest probability.
        #     return mu
    # def decoder(self, x: Variable) -> Variable:
    def decoder(self, x):
        x = self.fcnback(x)
        x = x.view([x.shape[0], 512, 1])
        x = self.ht(self.t_conv1(x))
        x = self.ht(self.t_conv2(x))
        x = self.ht(self.t_conv3(x))
        x = self.ht(self.t_conv4(x))
        x = self.ht(self.t_conv5(x))
        x = self.ht(self.t_conv6(x))
        x = self.ht(self.t_conv7(x))
        x = self.ht(self.t_conv8(x))
        return x


    # def forward(self, x: Variable)->(Variable, Variable, Variable):
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar



class VarAutoEncoder():

    def __init__(self,gauges=[702],
                 data_path='../data/_sjdf',
                 data_name='sjdf',
                 model_name='model0'):

        self.gauges = gauges                  #[702,712,901,902,911,912]
        self.ngauges = len(self.gauges)
        
        self.data_path = data_path

        if not os.path.exists('_output'):
            os.mkdir('_output')
        self.output_path = '_output'
        
        self.data_fname = None
        self.data_name = data_name

        self.shuffled = None
        self.shuffle_seed = 0
        self.init_weight_seed = 0

        self.model_name = model_name
        self.device='cpu'   # set to gpu if needed

        self.input_gauges_bool = None

        self.shuffled_batchno = False
        self.data_train_batch_list = None
        self.test_train_batch_list = None 

        self.use_Agg = True

        # set dpi for plt.savefig
        self._dpi = 300

    def load_data(self,
                  batch_size=20,
                  ngauges=3,
                  data_fname=None):
        '''
        load interpolated gauge data 

        set data_fname to designate stored data in .npy format
        
        '''

        device = self.device

        if data_fname == None:
            fname = self.model_name + '.npy'
            data_fname = os.path.join(self.data_path, fname)

        data_all = np.load(data_fname)
        self.nruns = data_all.shape[0]
        model_name = self.model_name
        data_name = self.data_name
        
        # load shuffled indices
        fname = os.path.join(self.data_path,
                            '{:s}_train_index.txt'.format(data_name))
        train_index = np.loadtxt(fname).astype(np.int)
        
        fname = os.path.join(self.data_path,
                            '{:s}_test_index.txt'.format(data_name))
        test_index = np.loadtxt(fname).astype(np.int)

        data_train = data_all[train_index, : , :]
        data_test  = data_all[test_index, : , :]

        # creat a list of batches for training, test sets
        data_train_batch_list = []
        data_test_batch_list = []

        self.batch_size = batch_size
        for i in np.arange(0, data_train.shape[0], batch_size):
            data0 = data_train[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_train_batch_list.append(data0)

        for i in np.arange(0, data_test.shape[0], batch_size):
            data0 = data_test[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_test_batch_list.append(data0)

        self.nbatches_train = len(data_train_batch_list)
        self.nbatches_test  = len(data_test_batch_list)

        self.data_train_batch_list = data_train_batch_list
        self.data_test_batch_list = data_test_batch_list
        
        self.data_fname = data_fname

    def train_vae(self,vaeruns=100,
                        zdims = 100,
                        torch_loss_func=nn.MSELoss,
                        torch_optimizer=optim.Adam,
                        nepochs=1000,
                        input_gauges=None,
                        top=[64, 32],
                        lr=0.0005):
        '''
        Set up and train autoencoder

        vaeruns :
            number of vae runs for the model

        rand_seed :
            seed for randomization 
            (for shuffling data and initializing weights)

        top :
            (times of prediction) mask data points after pts in time, 
            # of data pts in the uniformly-spaced time-series to use.  
        input_gauges :
            gauges to use as inputs, sublist of gauges

        torch_loss_func :
            pytorch loss function, default is torch.nn.MSELoss

        torch_optimizer :
            pytorch loss function, default is torch.nn.MSELoss

        '''

        # set random seed
        init_weight_seed = self.init_weight_seed
        torch.random.manual_seed(init_weight_seed)

        device = self.device

        data_train_batch_list = self.data_train_batch_list
        output_path = self.output_path
        ngauges = self.ngauges

        # select input gauge
        if input_gauges == None:
            # for default use only the first gauge
            input_gauges = self.gauges[:1]
        
        input_gauges = np.array(input_gauges)
        ninput = len(input_gauges)

        self.input_gauges = input_gauges
        input_gauges_bool = np.array(\
                [np.any(gauge == input_gauges) for gauge in self.gauges])
        self.input_gauges_bool = input_gauges_bool
        ig = np.arange(ngauges)[input_gauges_bool]

        model_name = self.model_name


        self.top = top
        self.vaeruns = vaeruns
        self.nepochs = nepochs
        self.device = device

        nbatches_train = len(data_train_batch_list)


        self.save_model_info()              # save model info
        save_interval = int(nepochs/10)    # save model every _ epochs

        for T in top:

            # define new model
            model = Conv1DVAE(ngauges,ninput, zdims)
            model.to(device)

            # train model
            loss_func = torch_loss_func
            optimizer = torch_optimizer(model.parameters(), lr=lr)

            # epochs
            train_loss_array = np.zeros(nepochs+1)

            for epoch in range(1, nepochs+1):
                # monitor training loss
                train_loss = 0.0

                #Training
                for k in range(nbatches_train):
                    data0 = data_train_batch_list[k]
                    optimizer.zero_grad()

                    # input is first two gauges, zero-ed out after T
                    data1 = data0[:,ig,:].detach().clone()
                    data1[:,:,T:] = 0.0

                    outputs = model(data1)
                    loss = loss_func(outputs[0], data0, outputs[1], outputs[2])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                        
                # avg training loss per epoch
                avg_train_loss = train_loss/nbatches_train
                train_loss_array[epoch] = avg_train_loss 

                msg = '\rT = {:4d}, epoch = {:4d}, loss = {:1.8f}'.format(T,epoch,avg_train_loss)
                sys.stdout.write(msg)
                sys.stdout.flush()

                fname = '{:s}_train_loss_{:02d}_{:02d}.npy'\
                        .format(model_name, T, 1)
                save_fname = os.path.join(output_path, fname)
                np.save(save_fname, train_loss_array)

                if ((epoch) % save_interval) == 0:
                    # save intermediate model
                    fname ='{:s}_{:02d}_{:02d}_{:04d}.pkl'\
                            .format(model_name, T, 1, epoch)
                    save_fname = os.path.join(output_path, fname)
                    torch.save(model, save_fname)



    def save_model_info(self):

        import pickle

        info_dict = [self.batch_size,
                     self.nbatches_train,
                     self.input_gauges,
                     self.input_gauges_bool,
                     self.nepochs,
                     self.vaeruns,
                     self.gauges,
                     self.ngauges,
                     self.top,
                     self.data_path,
                     self.output_path,
                     self.shuffled,
                     self.shuffle_seed,
                     self.init_weight_seed,
                     self.shuffled_batchno,
                     self.model_name,
                     self.data_fname,
                     self.device]
        
        fname = '{:s}_info.pkl'.format(self.model_name)
        save_fname = os.path.join(self.output_path, fname)
        pickle.dump(info_dict, open(save_fname,'wb'))
        

    def load_model(self,model_name,device=None):

        import pickle

        # load model info
        fname = '{:s}_info.pkl'.format(model_name)
        load_fname = os.path.join(self.output_path, fname)
        info_dict = pickle.load(open(load_fname,'rb'))

        [self.batch_size,
         self.nbatches_train,
         self.input_gauges,
         self.input_gauges_bool,
         self.nepochs,
         self.vaeruns,
         self.gauges,
         self.ngauges,
         self.top,
         self.data_path,
         self.output_path,
         self.shuffled,
         self.shuffle_seed,
         self.init_weight_seed,
         self.shuffled_batchno,
         self.model_name,
         self.data_fname,
         self.device] = info_dict

        if device != None:
            self.device = device

        # load data
        self.load_data(batch_size=self.batch_size,
                       ngauges=self.ngauges,
                       data_fname=self.data_fname)


    def predict_dataset(self,epoch, device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Notes
        -----
        the prediction result is stored as binary numpy arrays in 
        the output directory

        """

        # load data and data dimensions
        batch_size = self.batch_size

        data_train_batch_list = self.data_train_batch_list
        data_test_batch_list = self.data_test_batch_list

        nbatches = len(data_train_batch_list) \
                 + len(data_test_batch_list)
        
        ndata_train = sum([data0.shape[0] for 
                           data0 in data_train_batch_list])
        ndata_test  = sum([data0.shape[0] for 
                           data0 in data_test_batch_list])

        data_batch_list = data_train_batch_list \
                        + data_test_batch_list 
            
        ndata = ndata_train + ndata_test

        vaeruns = self.vaeruns
        top = self.top

        gauges = np.array(self.gauges)      
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        npts = data_batch_list[0].shape[-1] # TODO: store this somewhere?

        t_unif = np.linspace(0.0, 4.0, npts)

        device = self.device

        model_name = self.model_name
        if epoch == 0:
            epoch = self.nepochs     # use final epoch
        
        # predict dataset
        for T in top:
            pred_all = np.zeros((vaeruns, ndata, ngauges, npts))

            for n in range(vaeruns):
                model = self.eval_model(T, 1, epoch, device=device)

                k1 = 0

                for k in range(nbatches):  
                    msg = '\rTest set, T={:6d}, model={:6d}, batch={:6d}'\
                          .format(T,n,k)
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                    data0 = data_batch_list[k].to(device)

                    # setup input data
                    datak = data0[:,input_gauges_bool,:].detach().clone()
                    datak[:,:,T:] = 0.0
                    datak = datak

                    kbatch_size = datak.shape[0]

                    # evaluate model (TODO: switch model to eval mode)
                    # dims = data0.detach().numpy().shape
                    # mo = np.zeros((dims[0], dims[1], dims[2], vaeruns))
                    # for l in range(vaeruns):
                    #     model_out = model(datak)
                    #     mo[:,:,:, l] = model_out[0].detach().numpy()

                    # data0 = data0.detach().numpy()
                    # model_out = np.mean(mo[:,:,:,:], axis = 3)

                    # evaluate model
                    model_out = model(datak)
                    model_out = model_out[0].detach().numpy()
                    
                    # collect pred & obs
                    pred_all[n, k1:(k1 + kbatch_size), :, :] = model_out

                    k1+=kbatch_size
                    

            fname = '{:s}_{:s}_{:02d}_{:04d}.npy'\
                     .format(model_name, 'test', T, epoch)
            save_fname = os.path.join('_output', fname)
            np.save(save_fname, pred_all)



    def eval_model(self, T, n, epoch, device='cpu'):
        r"""
        Returns autoencoder model in evaluation mode

        Parameters
        ----------

        T : int
            prediction time, later values will be masked with zeros

        n : int
            model number in the ensemble
        
        epoch : int
            use model saved after specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        
        model : Conv1d module
            Conv1d module in evaluation mode

        """
        model_name = self.model_name

        # load stored autoencoder
        fname = '{:s}_{:02d}_{:02d}_{:04d}.pkl'\
                .format(model_name, T, n, epoch)

        load_fname = os.path.join('_output', fname)
        model = torch.load(load_fname,
                           map_location=torch.device(device))

        model.eval()

        return model



    def predict_input(self, model_input, epoch, device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        model_input : tensor
            model input of size (?, ninput_gauges, npts)

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        
        pred : list of arrays
            prediction results for each prediction time T, each item is a numpy
        array of the shape (nensemble, ?, ngauges, npts)

        Notes
        -----

        items in output pred is also saved to output directory in binary 
        numpy array format

        """

        # load data and data dimensions
        batch_size = self.batch_size

        vaeruns = self.vaeruns
        top = self.top

        gauges = np.array(self.gauges)      
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        ndata = model_input.shape[0]
        npts = model_input.shape[-1]

        model_input = torch.tensor(model_input, dtype=torch.float32)
        t_unif = np.linspace(0.0, 4.0, npts)    # TODO: store this elsewhere?

        device = self.device

        model_name = self.model_name
        if epoch == 0:
            epoch = self.nepochs     # use final epoch
        
        pred_all = []


        # predict dataset
        for T in top:

            pred = np.zeros((vaeruns, ndata, ngauges, npts))
            
            for n in range(vaeruns):

                model = self.eval_model(T, 1, epoch, device=device)

                data0 = model_input.to(device)

                # setup input data
                datak = data0[:, input_gauges_bool, :].detach().clone()
                datak[:, :, T:] = 0.0
                datak = datak

                # evaluate model
                model_out = model(datak)
                model_out = model_out[0].detach().numpy()

                # collect predictions
                pred[n, ...] = model_out

            # save output to _output
            fname = '{:s}_{:s}_{:02d}_{:02d}_{:04d}.npy'\
                     .format(model_name, 'input', T, n, epoch)
            save_fname = os.path.join('_output', fname)
            np.save(save_fname, pred)
            
            pred_all.append(pred)
        
        return pred_all


    def predict_eta_ts_allgauges(self, epoch=-1, out_format='png'):
        '''

        plot full time-series prediction for all gauges for fixed T
        for runs in the test set
        
        TODO: store / output results

        '''

        if self.use_Agg:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        top = self.top

        nensemble = self.nensemble
        model_name = self.model_name

        batch_size = self.batch_size

        # TODO: load this at load_model
        train_shuffled_runno = np.loadtxt('data/sjdf_train_runno.txt')
        test_shuffled_runno = np.loadtxt('data/sjdf_test_runno.txt')

        shuffled_runno = \
            np.hstack((train_shuffled_runno, test_shuffled_runno))

        data_train_batch_list = self.data_train_batch_list
        data_test_batch_list = self.data_test_batch_list

        nbatches = len(data_train_batch_list) \
                 + len(data_test_batch_list)
        
        ndata_train = sum([data0.shape[0] for 
                           data0 in data_train_batch_list])
        ndata_test  = sum([data0.shape[0] for 
                           data0 in data_test_batch_list])

        data_batch_list = data_train_batch_list \
                        + data_test_batch_list 
            
        ndata = ndata_train + ndata_test

        nt = data_batch_list[0].shape[-1]   # TODO: set nt elsewhere
        t_unif = np.linspace(0.0, 4.0, nt)  # TODO: set final time elsewhere

        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        gauges = self.gauges
        ngauges = self.ngauges

        device = self.device

        # use final epoch 
        if epoch == -1:
            epoch = self.nepochs - 1

        ii = 0
        for k in range(nbatches):
            for j in range(data_batch_list[k].shape[0]):
                orig_runno = shuffled_runno[ii]


                ii+=1
                # set T
                for T in top:
                    sys.stdout.write(\
           '\r plotting / runno ={:5f}, T={:5d}             '.format(orig_runno,T))
                    sys.stdout.flush()
                    data0 = data_batch_list[k].to(device)
                    datak_npy = data0[j:(j+1),:,:].detach().numpy() 

                    # set-up input data
                    datak = data0[j:(j+1),input_gauges_bool,:].detach().clone()
                    datak[0,:,T:] = 0.0

                    pred_all = np.zeros((nensemble,ngauges,nt))
                    tm = t_unif[T]

                    for nm in range(nensemble):
                        suffix = "_{:02d}_{:02d}_{:04d}.pkl".format(T,nm,epoch)
                        fname = os.path.join('_output',model_name + suffix)
                        model = torch.load(fname,
                                           map_location=torch.device(device))
                        model = torch.load(fname,
                                           map_location=torch.device('cpu'))

                        # evaluate model
                        # for l in range(vaeruns):
                        #     model_out = model(datak)
                        #     model_out_npy = model_out[0].detach().numpy()
                        #     pred_all[nm,l,:,:]= model_out_npy
                        # evaluate model
                        model_out = model(datak)
                        model_out_npy = model_out.detach().numpy()
                        pred_all[nm,:,:]= model_out_npy

                    sc = 1.1
                    # make plots
                    fig,axes = plt.subplots(ncols=1,nrows=ngauges,
                                            figsize=(sc*10,sc*ngauges*8/6),
                                            sharex=True)
                    color0 = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    # if nensemble==1:
                    #     pred_all = np.squeeze(pred_all)
                    # else:
                    #     break
                    # add vertical red line
                    vmax = 1.1*np.abs(pred_all[:,:,:]).max()
                    vmin = -vmax

                    th = np.linspace(vmin,vmax,2)
                    tw = tm + 0.0*th

                    for i in range(ngauges):
                        ax = axes[i]

                        if input_gauges_bool[i]:
                            T1 = T
                        else:
                            T1 = 0
                        
                        predm = np.mean(pred_all[:,i,T1:],axis=0)
                        pred2std = 2*np.std(pred_all[:,i,T1:],axis=0)
                            
                        line0,= ax.plot(t_unif[T1:],predm,
                                        color=color0[i])
                        
                        line1 = ax.fill_between(t_unif[T1:], 
                                predm-pred2std,predm+pred2std,
                                color=color0[i],
                                alpha=0.2,
                                linewidth=0.5)

                        # plot observed
                        line2,= ax.plot(t_unif[T1:], datak_npy[0,i,T1:].T,
                                        linestyle="--",
                                        color='k',
                                        linewidth=1.0)
                        ax.plot(t_unif[:T1], datak_npy[0,i,:T1].T,
                                    color='k',
                                    linewidth=1.0)

                        vmax = np.abs(datak_npy[0,i,T1:]).max()
                        vmin = - vmax
                        if input_gauges_bool[i]:
                            ax.plot(tw,th,"r",linewidth=1.0)

                        # add legend
                        gauge_no_str = "{:3d}".format(gauges[i])
                        ax.legend((line0,line2,line1),
                                  (gauge_no_str + " pred",
                                   gauge_no_str + " obs",
                                   r"$\hat{\mu} \pm 2 \hat{\sigma}$"),
                                   bbox_to_anchor=(1, 1), loc='upper left')
                        ax.set_ylim([vmin,vmax])
                                  #bbox_to_anchor=(1, 1), loc='upper left')

                    # set title
                    axes[0].set_title("run {:04f}, time of prediction {:02d} mins".format(orig_runno,tm))
                    axes[-1].set_xlabel("time (m)")
                    axes[-1].set_xlim([t_unif[0], t_unif[-1]])

                    fig.tight_layout()

                    # save file
                    suffix = model_name + "_{:05f}_{:04d}_{:04d}".format(orig_runno,T,epoch) + '.' + out_format
                    fname = "_plots/pred_" + suffix
                    fig.savefig(fname,dpi=self._dpi)
                    plt.close(fig)
        return True


    def predict_eta_ts_allT(self,out_format='png',epoch=-1):
        '''

        Plot full time-series prediction for all T for a fixed gauge

        TODO: store / output results

        '''

        if self.use_Agg:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # plot prediction for a fixed gauge, variable T (in the test set)
        if epoch == -1:
            # use final epoch 
            epoch = self.nepochs - 1

        t_unif = self.t_unif
        nt = len(t_unif)

        top = self.top
        ntop = len(top)

        nensemble = self.nensemble

        model_name = self.model_name
        gauges = self.gauges
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool
        
        nbatches_train = self.nbatches_train
        nbatches = self.nbatches
        batch_size = self.batch_size
        shuffled_batchno = self.shuffled_batchno 

        data_batch_list = self.data_batch_list 
        
        sc = 1.1    # figure size scale
        color0 = plt.rcParams['axes.prop_cycle'].by_key()['color']

        device = self.device


        for i in range(nbatches_train,nbatches):
            for j in range(batch_size):
                for k,gauge in enumerate(gauges):
                    orig_runno = j + 20*shuffled_batchno[i] 

                    fig,axes = plt.subplots(ncols=1,nrows=ntop,
                                            figsize=(sc*10,sc*ntop/6*8),
                                            sharex=True,sharey=True)
                    vmaxT = 0.0

                    for tn,T in enumerate(top):

                        sys.stdout.write(\
           '\r plotting / runno ={:5d}, T = {:4d}, gauge={:5d}        '.format(orig_runno,T,gauge))
                        if input_gauges_bool[k]:
                            T1 = T
                        else:
                            T1 = 0

                        data0 = data_batch_list[k].to(device)

                        # set-up input data
                        datak = data0[j:(j+1),input_gauges_bool,:].detach().clone()
                        datak[0,:,T:] = 0.0
                        datak_npy = data0[j:(j+1),:,:].detach().numpy() 

                        pred_all = np.zeros((nensemble,vaeruns, ngauges,nt))
                        tm = int(np.floor(t_unif[T]))
                        for nm in range(nensemble):
                            suffix = "_{:02d}_{:02d}_{:04d}.pkl".format(T,nm,epoch)
                            fname = os.path.join('_output', model_name + suffix)
                            model = torch.load(fname,map_location=torch.device(device))

                            # evaluate model
                            for l in range(vaeruns):
                                model_out = model(datak)
                                model_out_npy = model_out[0].detach().numpy()
                                pred_all[nm,l,:,:]= model_out_npy

                        # make plots
                        if nensemble==1:
                            pred_all = np.squeeze(pred_all)
                        else:
                            break
                        # add vertical red line
                        vmax = 1.1*np.abs(pred_all[:,:,:]).max()
                        vmaxT = max([vmax,vmaxT])
                        vmin = -vmax
                        vminT = -vmaxT

                        th = np.linspace(2.0*vmin,2.0*vmax,2)
                        tw = tm + 0.0*th
                        if ntop ==1:
                            ax = axes
                        else: 
                            ax = axes[ntop-tn-1]
                        ax.set_title('run {:04d}, gauge {:03d}, time of prediction {:2d} min'.format(orig_runno, gauge, int(t_unif[T]/60.0)))

                        predm = np.mean(pred_all[:,k,T1:],axis=0)
                        pred2std = 2*np.std(pred_all[:,k,T1:],axis=0)

                        line1 = ax.fill_between(t_unif[T1:]/60.0, 
                                                predm-pred2std,predm+pred2std,
                                                color=color0[k],
                                                alpha=0.2,
                                                linewidth=0.5)
                        line0,= ax.plot(t_unif[T1:]/60.0, predm,
                                        color=color0[k])

                        # plot observed
                        line2,= ax.plot(t_unif[T1:]/60.0, datak_npy[0,k,T1:].T,
                                    linestyle="--",
                                    color='k',
                                    #color=color0[k],
                                    linewidth=1.0)
                        ax.plot(t_unif[:T1]/60.0, datak_npy[0,k,:T1].T,
                                    color='k',
                                    linewidth=1.0)
                    
                        # add legend
                        gauge_no_str = "{:3d}".format(gauge)

                    # set title
                    if ntop==1:
                        axes.set_xlabel("time (m)")
                    else:
                        axes[-1].set_xlabel("time (m)")

                    ax.legend((line0,line2,line1),
                              ("pred","obs",r"$\hat{\mu} \pm 2 \hat{\sigma}$"),
                              bbox_to_anchor=(1, 1), loc='upper left')
                    ax.set_ylim([vminT,vmaxT])

                    fig.tight_layout()

                    # save file
                    suffix = model_name + "_{:04d}_{:03d}_{:04d}".format(orig_runno,gauge,epoch) + '.' + out_format
                    fname = "_plots/predvT_" + suffix
                    fig.savefig(fname, dpi=self._dpi)
                    plt.close(fig)

        return True

