import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, class_split, save_excel, save_image2
from torchplus.datasets import PreProcessFolder
from piq import SSIMLoss

if __name__ == "__main__":
    batch_size = 512
    train_epoches = 100
    log_epoch = 4
    class_num = 530
    root_dir = "D:/log/paper5/logZZPMAIN.aeface"
    dataset_dir = "E:/datasets/FaceScrub"
    h = 64
    w = 64
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"FaceScrub ae 8192",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir, model_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([Resize((h, w)), RandomHorizontalFlip(), ToTensor()])

    priv_ds = PreProcessFolder(root=dataset_dir, transform=transform, batch_size=1024)

    priv_ds_len = len(priv_ds)

    priv_train_ds, priv_test_ds = random_split(
        priv_ds, [priv_ds_len * 6 // 7, priv_ds_len - priv_ds_len * 6 // 7]
    )

    train_ds = priv_train_ds
    test_ds = priv_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    # for evaluate
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )
    # for attack
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
        pin_memory=True,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    class FeatureExtracter(nn.Module):
        def __init__(self):
            super(FeatureExtracter, self).__init__()
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

        def forward(self, x: Tensor):
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
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            x = self.relu4(x)
            x = x.view(-1, 8192)
            return x

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.sigmod = nn.Sigmoid()

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.deconv5(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().train(False).to(output_device)
    myinversion = Inversion(8192).train(True).to(output_device)

    optimizer_fe = optim.Adam(
        feature_extractor.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    optimizer = optim.Adam(
        myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, (im, label) in enumerate(tqdm(test_dl, desc=f"epoch {epoch_id} ")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            optimizer_fe.zero_grad()
            feature8192 = feature_extractor.forward(im)
            rim = myinversion.forward(feature8192)
            ssim = SSIMLoss()(rim, im)
            loss = ssim
            loss.backward()
            optimizer_fe.step()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("ssim", ssim, epoch_id)
            save_image2(im.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image2(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(
                os.path.join(log_dir, f"feature_extractor_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(feature_extractor.state_dict(), f)
            with open(
                os.path.join(model_dir, f"myinversion_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myinversion.state_dict(), f)

        if epoch_id % log_epoch == 0:
            with torch.no_grad():
                myinversion.eval()
                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("priv ssim", ssimlossavg, epoch_id)

                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"aux")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    rim = myinversion.forward(feature8192)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("aux ssim", ssimlossavg, epoch_id)
    writer.close()
