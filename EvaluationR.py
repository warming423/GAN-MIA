import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy
from ClassifierNet import Eclassifier

if __name__ == '__main__':
    batch_size = 64
    train_epoches = 30
    log_epoch = 5
    class_num = 170
    root_dir = "/home/wangm/毕设/log/ER"
    dataset_dir1= "/opt/datasets/PubFig"
    dataset_dir2= "/opt/datasets/VGGFace2"
    dataset_dir3 =  "/opt/datasets/FaceScrub"
    Num1=20
    Num2=130
    Num3=20

    dataset_usps = "/opt/sambashare/USPS"
    dataset_mnist="./data"

    h = 64
    w = 64

    init = Init(seed=999, log_root_dir=root_dir,
                backup_filename=__file__, tensorboard=True, comment=f'Evaluation Recongnizer--Pubfig-VGGface2-FaceScrub')
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([
        Resize((h, w)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

    # mnist_train_ds = MNIST(root=dataset_mnist, train=True,
    #      transform=transform, download=True)
    # mnist_test_ds = MNIST(root=dataset_mnist, train=False,
    #                       transform=transform, download=True)
    
    # usps_train_ds= USPS(root=dataset_usps, train=True,
    #      transform=transform, download=True)
    # usps_test_ds = USPS(root=dataset_usps, train=False,
                #          transform=transform, download=True)

    data_pub=ImageFolder(root=dataset_dir1,transform=transform)
    data_vgg= ImageFolder(root=dataset_dir2, transform=transform)
    data_fs=ImageFolder(root=dataset_dir3,transform=transform)

    small_dir1 = os.listdir(dataset_dir1)[:Num1]# 计算每个文件夹下的图片数量
    pic_num1 = 0 
    for small_dir in small_dir1:
        small_dir_path = os.path.join(dataset_dir1, small_dir)
        pic_count = len(os.listdir(small_dir_path))
        pic_num1 += pic_count
    print(pic_num1)
    data_Pub = torch.utils.data.Subset(data_pub, range(pic_num1))

    small_dir2 = os.listdir(dataset_dir2)[:Num2]# 计算每个文件夹下的图片数量
    pic_num2 = 0 
    for small_dir in small_dir2:
        small_dir_path = os.path.join(dataset_dir2, small_dir)
        pic_count = len(os.listdir(small_dir_path))
        pic_num2 += pic_count
    print(pic_num2)
    data_VGG = torch.utils.data.Subset(data_vgg, range(pic_num2))

    small_dir3 = os.listdir(dataset_dir3)[:Num3]# 计算每个文件夹下的图片数量
    pic_num3 = 0 
    for small_dir in small_dir3:
        small_dir_path = os.path.join(dataset_dir3, small_dir)
        pic_count = len(os.listdir(small_dir_path))
        pic_num3 += pic_count
    print(pic_num3)
    data_FS = torch.utils.data.Subset(data_fs, range(pic_num3))

    Concat_train=ConcatDataset([data_Pub,data_VGG,data_FS])
    Concat_test=ConcatDataset([data_Pub,data_VGG,data_FS])

    test_data = DataLoader(dataset=Concat_test, batch_size=batch_size,
                         shuffle=False, num_workers=data_workers, drop_last=False)
    train_data = DataLoader(dataset=Concat_train, batch_size=batch_size,
                          shuffle=True, num_workers=data_workers, drop_last=True)

    myclassifier = Eclassifier().train(True).to(output_device)

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
            with open(os.path.join(log_dir, f'EvaluationR_{epoch_id}.pkl'), 'wb') as f:
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
