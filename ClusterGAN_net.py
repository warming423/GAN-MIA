import numpy as np
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn
import torch.nn.functional as F
import torch
from itertools import chain as ichain
from utils import tlog, softmax, initialize_weights, calc_gradient_penalty


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )

class Generator(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape):
        super(Generator, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (256, 8, 8)
        self.iels = int(np.prod(self.ishape))
        
        self.seq1=nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2)
        )
        self.seq2=nn.Sequential(
             torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2)
        )
        self.reshape=Reshape(self.ishape)
        self.seq3=nn.Sequential(
           nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2)
        )
        self.seq4=nn.Sequential(
           nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2)
        )
        self.seq5=nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )      
        initialize_weights(self)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_1=self.seq1(z)
        x_2=self.seq2(x_1)
        x_3=self.reshape(x_2)
        x_4=self.seq3(x_3)
        x_5=self.seq4(x_4)
        x_6=self.seq5(x_5)
        # Reshape for output
        x_gen = x_6.view(x_5.size(0), *self.x_shape)
        
        return x_gen


class Encoder(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, n_c):
        super(Encoder, self).__init__()

        self.name = 'encoder'
        self.channels = 3
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (256, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1,bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2,padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2,padding=1, bias=True),
            nn.LeakyReLU(0.2),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits
    

class Discriminator(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.name = 'discriminator'
        self.channels = 3
        self.cshape = (256, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        
        self.seq1=nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2,padding=1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.seq2=nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2,padding=1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.seq3=nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2,padding=1,bias=True),
            nn.LeakyReLU(0.2)
        )
        self.seq4=nn.Sequential(
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2)
        )
        self.seq5=nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
        
        initialize_weights(self)

    def forward(self, img):
        # Get output
        x=self.seq1(img)
        x=self.seq2(x)
        x=self.seq3(x)
        x=x.view(x.size(0),-1)
        x=self.seq4(x)
        x=self.seq5(x)
        validity = x
        
        return validity
