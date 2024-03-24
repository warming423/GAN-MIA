from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from DCGAN_net import Generator,Discriminator
from torchvision.utils import save_image
from torchplus.utils import Init


if __name__ == '__main__':

    manualSeed = 123
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataset_dir="/opt/datasets/FaceScrub"
    model_dir="/home/wangm/MIA/DCGAN-MIA/log/G/mini_model"
    log_dir = "/home/wangm/MIA/DCGAN-MIA/log/G/mini_picture"
    workers = 2 
    batch_size = 32
    num_epochs = 50
    h=64
    w=64
    z_dim=100
    real_label = 1.
    fake_label = 0.
    lr=0.0002
    beta1=0.5
    device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
    
    # 检查文件夹是否存在，不存在则创建
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 不存在，已成功创建。")
        else:
            print(f"文件夹 '{folder_path}' 已存在。")
    # 调用函数并传入文件夹路径
    create_folder_if_not_exists(model_dir)
    create_folder_if_not_exists(log_dir)


    transform = Compose([
        Resize((h, w)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
        #GaussianBlur(kernel_size=5, sigma=5)
    ])

    orig_set= ImageFolder(root=dataset_dir,transform=transform)
    dataloader = torch.utils.data.DataLoader(orig_set, batch_size=batch_size,shuffle=True, num_workers=workers,drop_last=True)

    #权重初始化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG = Generator().to(device) 
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    fixed_noise = torch.randn(batch_size, z_dim,device=device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0): 
        ##### (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            
            netD.zero_grad()
            real_cpu = data[0].to(device)   
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)   #128-dimension
            errD_real = criterion(output, label) # Calculate loss on all-real batch
            errD_real.backward()
            D_x = output.mean().item()  #output所有128元素的均值

            ## Train with all-fake batch
            noise = torch.randn(b_size,z_dim,device=device)
            fake = netG(noise) 
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

        ####### (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    fake_tensor=vutils.make_grid(fake,nrow=10,padding=2,normalize=True)
                    save_image(fake_tensor,fp=f"{log_dir}/CelebA_{epoch}_{iters}.png")
            iters += 1

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        with open(os.path.join(model_dir, f'CelebA_NetD_{epoch}.pkl'), 'wb') as f:
             torch.save(netD.state_dict(), f)
        with open(os.path.join(model_dir, f'CelebA_NetG_{epoch}.pkl'), 'wb') as f:
             torch.save(netG.state_dict(), f)





        
        