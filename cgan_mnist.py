import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy
torch.cuda.init()

##########         tensorboard --logdir="/home/wangm/MIA" --port=6007

if __name__ == '__main__':
    batch_size = 32
    train_epoches = 50
    log_epoch = 5
    class_num = 10
    root_dir = "/home/wangm/MIA/DCGAN-MIA/log/TarR/T-MNIST"
    dataset_dir1 =  "/opt/datasets/MNIST"
    h = 28
    w = 28

    init = Init(seed=1, log_root_dir=root_dir,
                backup_filename=__file__, tensorboard=True, comment=f'T with F ')
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([
        Resize((h, w)),
        ToTensor(), 
        Grayscale() 
    ])
    trainData = torchvision.datasets.MNIST(dataset_dir1,train = True,transform = transform,download=True)
    testData = torchvision.datasets.MNIST(dataset_dir1,train = False,transform = transform,download=True)
    
    train_data = torch.utils.data.DataLoader(dataset = trainData,batch_size = batch_size,shuffle = True,drop_last=False)
    test_data = torch.utils.data.DataLoader(dataset = testData,batch_size = batch_size,drop_last=True,shuffle=True)

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
    
    myclassifier = Classifier().train(True).cuda()

    #optimizer = optim.SGD(myclassifier.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001)
    optimizer = optim.Adam(myclassifier.parameters(), lr=0.0002,
                           betas=(0.5, 0.999), amsgrad=True)

    for epoch_id in tqdm(range(1, train_epoches+1), desc='Total Epoch'):
        for i, (images, label) in enumerate(tqdm(train_data, desc=f'epoch {epoch_id}')):
            images = images.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = images.shape

            optimizer.zero_grad()
            out = myclassifier.forward(images)
            ce = nn.CrossEntropyLoss()(out, label)
            loss = ce
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            train_ca = ClassificationAccuracy(class_num)
            after_softmax = F.softmax(out, dim=-1)
            predict = torch.argmax(after_softmax, dim=-1)
            train_ca.accumulate(label=label, predict=predict)
            acc_train = train_ca.get()
            writer.add_scalar('loss', loss, epoch_id)
            writer.add_scalar('acc_training', acc_train, epoch_id)
            with open(os.path.join(log_dir, f'{epoch_id}.pkl'), 'wb') as f:
                torch.save(myclassifier.state_dict(), f)
         
            with torch.no_grad():
                myclassifier.eval()
                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (images, label) in enumerate(tqdm(train_data, desc='testing train')):
                    r += 1
                    images = images.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = images.shape
                    out = myclassifier.forward(images)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss/r
                acc_test = test_ca.get()
                writer.add_scalar('train loss', celossavg, epoch_id)
                writer.add_scalar('acc_train', acc_test, epoch_id)

                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (images, label) in enumerate(tqdm(test_data, desc='testing test')):
                    r += 1
                    images = images.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = images.shape
                    out = myclassifier.forward(images)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss/r
                acc_test = test_ca.get()
                writer.add_scalar('test loss', celossavg, epoch_id)
                writer.add_scalar('acc_test', acc_test, epoch_id)

    writer.close()

