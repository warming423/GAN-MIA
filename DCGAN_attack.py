import torch, os, time, random,utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import DCGAN_net
from tqdm import tqdm
import ClassifierNet
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.transforms.functional import *

log_dir="/home/wangm/MIA/DCGAN-MIA/log/result"
z_dim=100

device = torch.device("cuda:1")
# 确保 PyTorch 使用指定的设备
torch.cuda.set_device(device)

def gen_points_on_sphere(current_point, points_count, sphere_radius):
	points_shape = (points_count,) + current_point.shape
	perturbation_direction = torch.randn(*points_shape).cuda()
	dims = tuple([i for i in range(1, len(points_shape))])

	perturbation_direction = (sphere_radius/ torch.sqrt(torch.sum(perturbation_direction ** 2, axis = dims, keepdims = True))) * perturbation_direction
	sphere_points = current_point + perturbation_direction

	return sphere_points, perturbation_direction


def is_target_class(idens, target, T):
	val_iden = decision(idens,T)
	val_iden[val_iden != target] = -1
	val_iden[val_iden == target] = 1
	return val_iden

#######返回通过生成器之后的数据的类别
def decision(imgs, T):
	#imgs = utils.low2high(imgs)
	with torch.no_grad():
		T_out = T(imgs)
		val_iden = torch.argmax(T_out, dim=1).view(-1)      
	return val_iden 
		
#####返回与攻击身份相同标签的点
def gen_initial_points(attack_iden,G, T, min_clip, max_clip, z_dim):
	search_times=0
	progress_bar = tqdm(desc='Searching', unit='iterations')
	with torch.no_grad():
		while True:
			initial_points = torch.randn(1, z_dim).cuda().float().clamp(min=min_clip, max=max_clip)
			gen_img = G(initial_points)
			gen_label= decision(gen_img, T)
			search_times+=1
			#print("尝试寻找中,尝试次数:{:.2f}\t".format(search_times))
			if gen_label == attack_iden:
				break
			else:
				progress_bar.update(1)
				continue
		progress_bar.close()
	return initial_points

def inversion(hard_label,G,T,MaxIter=5000,N=32,current_sphere_radius=2,gama=1.3,step_size=3,min_clip=-1.5,max_clip=1.5,iden_start=10):
	
	G.cuda().eval()
	TarR.cuda().eval()
	iden=iden_start
	
	for i in range(num_labels):
    
		current_sphere_radius=2
		iters=0
		attack_iden=hard_label[i]
		iden+=1
		tf=time.time()
		#attack_iden_tensor = torch.tensor([attack_iden]).cuda()
		current_point=gen_initial_points(attack_iden,G,T,min_clip,max_clip,z_dim)
		interval = time.time() - tf	
		print("找到初始化点所需的时间:{:.2f}\t".format(interval))
		
		with torch.no_grad():
			initial_img=G(current_point).detach().cpu()
			show_initial_img=vutils.make_grid(initial_img,normalize=True)
			save_image(show_initial_img,fp=f"{log_dir}/initial_{iden}.png")

		for iters in tqdm(range(MaxIter),desc="Iter times"):
	
			sphere_points,perturbation_directions=gen_points_on_sphere(current_point,N,current_sphere_radius)
			new_points_classification = is_target_class(G(sphere_points.squeeze(1)),attack_iden,T)

			if new_points_classification.sum() > 0.75 * N :# == attack_params['sphere_points_count']:
				current_sphere_radius = current_sphere_radius * gama
				#final_point=current_point
			else:
				new_points_classification = (new_points_classification - 1)/2
				grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius
				change_value = torch.mean(grad_direction, dim=0, keepdim=True)

				current_point_new = current_point + step_size * change_value
				current_point_new = current_point_new.clamp(min_clip,max_clip)
				if decision(G(current_point_new),T) == attack_iden :
					current_point=current_point_new
				else:
					continue
			iters+=1
			if iters % 100 == 0 :
				with torch.no_grad():
					gen_img=G(current_point).detach().cpu()
					show_img=vutils.make_grid(gen_img,normalize=True)
					save_image(show_img,fp=f"{log_dir}/{iden}_{iters}_{current_sphere_radius}.png")

if __name__ == '__main__':

	tarR_path = "/home/wangm/MIA/DCGAN-MIA/log/TarR/2023Nov28_22-12-52_aisec-dell-server_FaceScrub and PubFig--256/PubFig_50.pkl"
	g_path = "/home/wangm/MIA/DCGAN-MIA/log/G/model/NetG_19.pkl"
	
	TarR = ClassifierNet.TClassifier(num_classes=150)
	TarR.load_state_dict(torch.load(tarR_path))
	G=DCGAN_net.Generator()
	G.load_state_dict(torch.load(g_path))
 
	num_labels=20
	iden_start=100
	hard_labels=torch.zeros(num_labels)
	for i in range(num_labels):
		hard_labels[i]=i+iden_start
  
inversion(hard_labels,G,TarR,iden_start=iden_start)