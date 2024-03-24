import torch
import torch.nn as nn

z_dim=100

class Generator(nn.Module):
    def __init__(self,SizeofFeature=z_dim):
        super(Generator, self).__init__()
        self.convt1=nn.Sequential(
            nn.ConvTranspose2d(SizeofFeature,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.convt2=nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.convt3=nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.convt4=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.convt5=nn.Sequential(
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        out=self.convt1(x)
        out=self.convt2(out)   
        out=self.convt3(out)
        out=self.convt4(out)        
        out=self.convt5(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )            
        self.conv4=nn.Sequential(
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(512,1,4,1,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.conv5(out)

        return out






