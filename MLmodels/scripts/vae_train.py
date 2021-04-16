r"""
Train variational autoencoder

"""
import os
import numpy as np
import vae
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable

# def loss_function(recon_x, x, mu, logvar) -> Variable:
def loss_function(recon_x, x, mu, logvar):
    # how well do input x and output recon_x agree?
    BCE = F.mse_loss(recon_x, x)
    # BCE = F.smooth_l1_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= (recon_x.shape[0] * (recon_x.shape[1]*recon_x.shape[2])) #was just sum of dimensions/number of entries
    #KLD /= recon_x.shape[0] * recon_x.shape[-1]
    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    # print(BCE.detach().numpy(), KLD.detach().numpy())
    return BCE + KLD

if __name__ == "__main__":

    VAE = vae.VarAutoEncoder(gauges=[702,901,911],\
                         model_name='vae_sjdf')

    # use GPU for training (e.g. on cluster)
    VAE.device = 'cuda'

    # load data
    VAE.load_data(batch_size=150,data_fname='../data/_sjdf/sjdf.npy') 

    # train ensemble
    VAE.train_vae(vaeruns=100,
                    zdims=450,
                    torch_loss_func=loss_function,
                    torch_optimizer=optim.Adam,
                    nepochs=1250,  
                    input_gauges=[702],
                    top=[51, 26])  

