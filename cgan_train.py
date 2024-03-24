from __future__ import print_function
import os
import random
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from ClusterGAN_net import Generator, Discriminator, Encoder
from torchvision.utils import save_image
from utils import save_model, calc_gradient_penalty, sample_z, cross_entropy,plot_train_loss
from itertools import chain as ichain

if __name__ == '__main__':

    manualSeed = 666
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataset_dir = "/home/wangm/MIA/DCGAN-MIA/log/111"
    model_dir = "/home/wangm/MIA/DCGAN-MIA/log/CG/model/20-face"
    log_dir = "/home/wangm/MIA/DCGAN-MIA/log/CG/picture/20-face"
    workers = 2
    batch_size = 64
    num_epochs = 500
    image_size = 64
    channels = 3
    image_shape = (channels, image_size, image_size)
    
    z_n = 100
    z_c = 20
    betan = 10
    betac = 10
    
    n_skip_iter = 1
    lr1=0.0001
    lr2 = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    decay = 2.5*1e-5

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

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
        Resize((image_size, image_size)),
        ToTensor(),
        #Grayscale(),
        Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ]) 

    orig_set = ImageFolder(root=dataset_dir, transform=transform)
    ds_len = len(orig_set)
    print(ds_len)
    train_len = int(0.9 * ds_len)
    test_len = ds_len - train_len
    train_set, test_set = random_split(orig_set, [train_len, test_len])

    train_data = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers,
                                             drop_last=True)
    test_data = torch.utils.data.DataLoader(test_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=workers,
                                            drop_last=False)
    
    test_imgs, test_labels = next(iter(test_data))
    test_imgs = test_imgs.cuda()
    test_imgs.requires_grad = True

    generator = Generator(z_n, z_c, image_shape)
    encoder = Encoder(z_n, z_c)
    discriminator = Discriminator()
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()

    ge_chain = ichain(generator.parameters(), encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain,
                                    lr=lr1,
                                    betas=(beta1, beta2),
                                    weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=lr2,
                                   betas=(beta1, beta2))

    ge_l = []
    d_l = []
    c_zn = []
    c_zc = []
    c_i = []
    
    torch.autograd.set_detect_anomaly(True)
    print('\nBegin training session with %i epochs...\n' % (num_epochs))
    
    for epoch in range(num_epochs):
        for i, (imgs, itruth_label) in enumerate(train_data):

            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()
            discriminator.train()

            real_imgs = imgs.to(device)  
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=z_n, n_c=z_c)
            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GE.zero_grad()
            # Encode the generated images   
            gen_imgs = generator(zn, zc)
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)  
            
            # Calculate losses for z_n, z_c
            zn_loss = mse_loss(enc_gen_zn, zn)
            zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
         
            # Wasserstein GAN loss
            ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
            ge_loss.backward()      
            optimizer_GE.step()
           
                      
            optimizer_D.zero_grad()  
            grad_penalty = calc_gradient_penalty(discriminator, real_imgs,gen_imgs.detach())
            # Wasserstein GAN loss w/gradient penalty
            d_loss = torch.mean(D_real) - torch.mean(discriminator(gen_imgs.detach())) + grad_penalty
            d_loss.backward()                    
            optimizer_D.step()  
                                        
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())

        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        ## Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        #r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]                                   
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        # Generate sample instances from encoding
        teg_imgs = generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())

        ## Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=z_n,
                                                 n_c=z_c)
        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)
        # Encode sample instances
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)
        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        #lat_xe_loss = cross_entropy(zc_e_logits, zc_samp)
        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())

        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        save_image(r_imgs.data[:n_samp],
                   '%s/real_%06i.png' %(log_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp],
                   '%s/reg_%06i.png' %(log_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp],
                   '%s/gen_%06i.png' %(log_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        
        # stack_imgs = []
        # for idx in range(z_c):
        #     # Sample specific class
        #     zn_samp, zc_samp, zc_samp_idx = sample_z(shape=z_c,
        #                                              latent_dim=z_n,
        #                                              n_c=z_c,
        #                                              fix_class=idx)

        #     # Generate sample instances
        #     gen_imgs_samp = generator(zn_samp, zc_samp)

        #     if (len(stack_imgs) == 0):
        #         stack_imgs = gen_imgs_samp
        #     else:
        #         stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)
                
        # Save class-specified generated examples!
        # save_image(stack_imgs,
        #            '%s/gen_classes_%06i.png' %(log_dir, epoch), 
        #            nrow=z_c, normalize=True)
     

        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     num_epochs, 
                                                     d_loss.item(),
                                                     ge_loss.item())
              )
        
        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]"%(img_mse_loss.item(), 
                                                             lat_mse_loss.item(), 
                                                             lat_xe_loss.item())
             )
   # Save training results
    train_df = pd.DataFrame({
                             'n_epochs' : num_epochs,
                             'learning_rate' : lr2,
                             'beta_1' : beta1,
                             'beta_2' : beta2,
                             'weight_decay' : decay,
                             'n_skip_iter' : n_skip_iter,
                             'latent_dim' : z_n,
                             'n_classes' : z_c,
                             'beta_n' : betan,
                             'beta_c' : betac,
                             'gen_enc_loss' : ['G+E', ge_l],
                             'disc_loss' : ['D', d_l],
                             'zn_cycle_loss' : ['$||Z_n-E(G(x))_n||$', c_zn],
                             'zc_cycle_loss' : ['$||Z_c-E(G(x))_c||$', c_zc],
                             'img_cycle_loss' : ['$||X-G(E(x))||$', c_i]
                            })

    train_df.to_csv('%s/training_details.csv'%(model_dir))
     # Plot some training results
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png'%(model_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
                    figname='%s/training_cycle_loss.png'%(model_dir)
                    )

    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=model_dir)