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
from ClassifierNet import TClassifier,FaceNet64,IR152
torch.cuda.init()

##########         tensorboard --logdir="/home/wangm/MIA" --port=6007

if __name__ == '__main__':
    batch_size = 32
    train_epoches = 50
    log_epoch = 5
    class_num = 530
    root_dir = "/home/wangm/MIA/DCGAN-MIA/log/TarR"
    dataset_dir1 =  "/opt/datasets/FaceScrub"
    h = 64
    w = 64
    image_size=64

    init = Init(seed=1, log_root_dir=root_dir,
                backup_filename=__file__, tensorboard=True, comment=f'T with F ')
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([
        Resize((h, w)),
        ToTensor(),   
        Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))       
    ])

    data_FS = ImageFolder(root=dataset_dir1,transform=transform)
    ds_len=len(data_FS)
    print(ds_len)
    train_len=int(0.8*ds_len)
    test_len=ds_len-train_len

    train_set, test_set = random_split(data_FS,[train_len, test_len])
    train_data = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True, num_workers=data_workers, drop_last=True)
    test_data = DataLoader(dataset=test_set, batch_size=batch_size,
                         shuffle=False, num_workers=data_workers, drop_last=False)
    
    myclassifier = TClassifier(num_classes=class_num).train(True).cuda()

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

