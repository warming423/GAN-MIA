import torch.nn as nn
import torch

z_dim=100
batch_size=32


class FeatureExtracter(nn.Module):
            def __init__(self):
                super(FeatureExtracter, self).__init__()
                self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
                self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
                self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
                self.bn1 = nn.BatchNorm2d(128)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(512)
                self.mp1 = nn.MaxPool2d(2, 2)
                self.mp2 = nn.MaxPool2d(2, 2)
                self.mp3 = nn.MaxPool2d(2, 2)
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()
                self.relu3 = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.mp1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.mp2(x)
                x = self.relu2(x)
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.mp3(x)
                x = self.relu3(x)
                x = x.view(batch_size, -1)
                return x

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)

    def forward(self, x):
        feature = self.bottleneck(x)
        out = self.fc(feature)
        return out
        
class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.feature_extractor = FeatureExtracter()
        self.cls = CLS(4608, 10, bottle_neck_dim=50)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cls(x)
        return x
    
    
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
                nn.ConvTranspose2d(64,1,4,2,1,bias=False),
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
            nn.Conv2d(1,64,4,2,1,bias=False),
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