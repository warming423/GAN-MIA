import torch
import torch.nn as nn
import evolve

batch_size=32
target_classifier_num=150
evl_num=170
dig_num=10

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
#############人脸识别模型##############
class TClassifier(nn.Module):
        def __init__(self, num_classes=target_classifier_num):
            super(TClassifier, self).__init__()
            self.out_features = num_classes
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)          
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.mp4 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.fc1 = nn.Linear(8192, num_classes*2)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(num_classes*2, self.out_features)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.mp2(x)     
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu2(x)
            x = self.mp3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.mp4(x)
            x = x.view(-1, 8192)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x


class FaceNet64(nn.Module):
    def __init__(self, num_classes=target_classifier_num):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out

class IR152(nn.Module):
    def __init__(self, num_classes=target_classifier_num):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out
 
##################评估模型######################
class FeatureBranch(nn.Module):
    def __init__(self):
        super(FeatureBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1)
        self.conv5 = nn.Conv2d(32, 64, 2, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        return x

class AttentionBranch(nn.Module):
    def __init__(self) -> None:
        super(AttentionBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 1, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()
        self.lrelu3 = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

class LastBranch(nn.Module):
    def __init__(self,class_num=evl_num):
        super(LastBranch, self).__init__()
        middle_num=class_num*2
        self.conv1 = nn.Conv2d(64, 128, 4, 2)
        self.conv2 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(1152, middle_num)
        self.fc2 = nn.Linear(middle_num, class_num)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x.view(-1, 1152)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class Eclassifier(nn.Module):
    def __init__(self) -> None:
        super(Eclassifier, self).__init__()
        self.fbranch = FeatureBranch()
        self.abranch = AttentionBranch()
        self.lbranch=LastBranch()

    def forward(self, x):
        f = self.fbranch(x)
        a = self.abranch(x)
        z=f*a
        out=self.lbranch(z)

        return out
