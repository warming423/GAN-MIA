import torch, os, time, random,utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import cgan_net
import ClusterGAN_net
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.transforms.functional import *
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import conditionalGAN_net


if __name__ == '__main__':
    
	image_size = 28
	channels = 1
	image_shape = (channels, image_size, image_size)    
	zn_dim = 30
	zc_dim = 10
	label_num=10
	each_label_vector_len=100
	resize=transforms.Resize([28,28],antialias=True)
 
	logdir="/home/wangm/MIA/DCGAN-MIA/log/cgan/mnist"
	tarR_path = "/home/wangm/MIA/DCGAN-MIA/log/TarR/T-MNIST/2024Feb25_20-29-25_aisec-dell-server_T with F /40.pkl"
	g_path = "/home/wangm/MIA/DCGAN-MIA/log/G/mnist_model/CelebA_NetG_25.pkl"
	cg_path= "/home/wangm/MIA/DCGAN-MIA/log/CG/model/generator.pth.tar"
	ce_path="/home/wangm/MIA/DCGAN-MIA/log/CG/model/encoder.pth.tar"
	condition_path="/home/wangm/MIA/model/MNIST/gen9.pkl"
	device = torch.device("cuda:1" if (torch.cuda.is_available() ) else "cpu")

	TarR = cgan_net.Classifier()
	TarR.load_state_dict(torch.load(tarR_path))
	G=cgan_net.Generator()
	G.load_state_dict(torch.load(g_path))
	# CG=ClusterGAN_net.Generator(zn_dim, zc_dim, image_shape)
	# CG.load_state_dict(torch.load(cg_path))
	# CE=ClusterGAN_net.Encoder(zn_dim, zc_dim)
	# CE.load_state_dict(torch.load(condition_path))
 
	conG=conditionalGAN_net.Generator(10,10,5,1,64)
	conG.load_state_dict(torch.load(condition_path))
 
	def zc_generation(bs=32,dim=10):
		z_c=torch.Tensor(bs,dim).fill_(0).to(device)
		for i in range(bs):
			z_c[i,random.randint(0,label_num-1)]=1
		return z_c
	'''	
	vanlliaGAN tsne
	vec_recoder=[]
	for label in range(10):
     
		vec_len=0
		G.eval().to(device)
		TarR.eval().to(device)
		while(vec_len <100):
			random_vector=torch.randn(cgan_net.batch_size,100,device=device)
			fake_image=G(random_vector)
			fake_image=resize(fake_image)
			fake_image=fake_image.view(cgan_net.batch_size,1,-1,28)
			output=TarR(fake_image)
			output=F.softmax(output,dim=-1)
			iden=torch.argmax(output,dim=-1)

			for i in range(cgan_net.batch_size):
				if iden[i]==label and vec_len<100:
					vec_recoder.append(random_vector[i])
					vec_len+=1
	print(len(vec_recoder))
	'''
	vec_recoder=[]
	for label in range(10):		
		vec_len=0
		conG.eval().to(device)
		TarR.eval().to(device)
		while(vec_len <100):
			random_vector=torch.randn(conditionalGAN_net.BATCH_SIZE,conditionalGAN_net.Z_DIM,device=device)
			label_vector= torch.randint(0, 10, (conditionalGAN_net.BATCH_SIZE,),dtype=int).to(device)
			#label_vector=torch.zeros((conditionalGAN_net.BATCH_SIZE,),dtype=int).fill_(random_int).to(device)
			fake_image=conG(random_vector,label_vector)
			fake_image=resize(fake_image)
			fake_image=fake_image.view(conditionalGAN_net.BATCH_SIZE,1,-1,28)
			output=TarR(fake_image)
			output=F.softmax(output,dim=-1)
			iden=torch.argmax(output,dim=-1)

			for i in range(conditionalGAN_net.BATCH_SIZE):
				if iden[i]==label and vec_len<100:
					vec_recoder.append(random_vector[i])
					vec_len+=1
					print(vec_len)
	print(len(vec_recoder))
	print("1111")
	'''
	ClusterGAN tsne
	vec_recoder=[]
	for label in tqdm(range(0,label_num),desc="fprocess"):
     
		vec_len=0
		G.eval().to(device)
		CG.eval().to(device)
		CE.eval().to(device)
		TarR.eval().to(device)
  
		while(vec_len <each_label_vector_len):
			z_n=0.75*torch.randn(cgan_net.batch_size,zn_dim,device=device)
			z_c=zc_generation(cgan_net.batch_size,zc_dim)
			fake_image=CG(z_n,z_c)			
			output=TarR(fake_image)
			output=F.softmax(output,dim=-1)
			iden=torch.argmax(output,dim=-1)
			enc_zn, enc_zc, enc_zc_logits = CE(fake_image)
			cat_vec=torch.cat((enc_zn.detach(),enc_zc_logits.detach()),dim=1)
			cat_vec.requires_grad=False

			for i in range(cgan_net.batch_size):
				if iden[i]==label and vec_len<each_label_vector_len:				
					vec_recoder.append(cat_vec[i])
					vec_len+=1
	print(len(vec_recoder))
	'''
 
	img_lists=[]
	for i in range(0,len(vec_recoder)):
		label=torch.tensor([[i//100]]).to(device)
		
		show_pic=conG(vec_recoder[i].unsqueeze(0),label).cpu()
		#show_pic=CG(vec_recoder[i][:zn_dim].unsqueeze(0),vec_recoder[i][zn_dim:].unsqueeze(0)).cpu()
		img_lists.append(show_pic)
	fig,axes=plt.subplots(label_num,len(img_lists)//label_num,figsize=(10*label_num,label_num))
	for i,tensors in enumerate(img_lists):
		images=vutils.make_grid(tensors,nrow=1,normalize=True)
		axes[i//each_label_vector_len,i%each_label_vector_len].imshow(np.transpose(images, (1, 2, 0)))
		axes[i//each_label_vector_len,i%each_label_vector_len].axis("off")
		#imgs=vutils.make_grid(img_lists,nrow=100,padding=2,normalize=True)
		#img_lists.append(vutils.make_grid(show_pic, nrow=1,padding=2, normalize=True))
  
	plt.show()
	plt.savefig(f"{logdir}/conGAN-sample.png")
 	
	print("sss")
	#[1,1 ... 2,2  3,3 ... 9,9]num ofeach integer will be 100 
	labels = np.repeat(range(label_num), each_label_vector_len)
	# Color and marker for each true class	
	colors = plt.cm.rainbow(np.linspace(0, 1, label_num))
	markers = matplotlib.markers.MarkerStyle.filled_markers
 
	numpy_tensors = [tensor.cpu() for tensor in vec_recoder]
	img_np=np.array(numpy_tensors)
	# 使用t-SNE进行降维
	tsne = TSNE(n_components=2, random_state=42,perplexity=10)
	embedded_vectors = tsne.fit_transform(img_np)

	# 绘制散点图
	plt.figure(figsize=(8, 8))
	for i in range(label_num):
		indices = labels == i
		plt.scatter(embedded_vectors[indices, 0], embedded_vectors[indices, 1], color=colors[i], label=str(i))
	plt.show()
	plt.savefig(f"{logdir}/conGAN-tsne.png")
