# 数据集每个身份中一张图片
'''
import os
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image

def read_image(image_path):
    image = Image.open(image_path)
    return image

big_folder_path = "/opt/datasets/FaceScrub"
small_folder_list = os.listdir(big_folder_path)

for small_folder in small_folder_list:
    small_folder_path = os.path.join(big_folder_path, small_folder)
    image_list = os.listdir(small_folder_path)
    for i, image_name in enumerate(image_list):
        if i == 0:
            image_path = os.path.join(small_folder_path, image_name)
            image = read_image(image_path)
            tensor_image = transforms.ToTensor()(image)
            save_image(tensor_image, f"/home/wangm/dataset/FaceScurb/{small_folder}.png")
            break

'''

''' 转灰度图
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# 定义数据转换
transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图
    transforms.ToTensor()   # 转换为 tensor
])

dataset = ImageFolder(root="/opt/datasets/PubFig", transform=transform)

for i, (image, label) in enumerate(dataset):
    image = transforms.ToPILImage()(image)
    filename = f'/home/wangm/毕设/log/convention/grey/{i}.jpg'
    image.save(filename)
'''
'''
from icecream import ic
from torchvision import datasets
from tqdm import tqdm
import os


train_data = datasets.USPS(root="/opt/sambashare/USPS", train=True, download=True)
test_data = datasets.USPS(root="/opt/sambashare/USPS", train=False, download=True)
saveDirTrain = './USPSImages-Train'
saveDirTest = './USPSImages-Test'

if not os.path.exists(saveDirTrain):
    os.mkdir(saveDirTrain)
if not os.path.exists(saveDirTest):
    os.mkdir(saveDirTest)

ic(len(train_data), len(test_data))
ic(train_data[0])
ic(train_data[0][0])


def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))


save_img(train_data, saveDirTrain)
save_img(test_data, saveDirTest)
'''
'''
import DCGAN_net
import torch
import torchvision.utils as vutils
import torch, os, time, random,utils
from torchvision.utils import save_image

g_path = "/home/wangm/MIA/DCGAN-MIA/log/G/model/NetG_5.pkl"

G=DCGAN_net.Generator()
G.load_state_dict(torch.load(g_path))
log_dir="/home/wangm/MIA/DCGAN-MIA/log/result"

for i in range(10):
    initial_points = torch.randn(1,100).cuda().float()
    G.cuda()
    show_img=G(initial_points).detach().cpu()
    #show = utils.low2high(show_img).squeeze(0)
    show=vutils.make_grid(show_img,normalize=True)
    save_image(show,fp=f"{log_dir}/{i}.png")
'''
'''
import os

# 大文件夹的路径
big_folder_path = '/home/wangm/MIA/DCGAN-MIA/log/FaceScrub'

# 获取大文件夹下的所有子文件夹
subfolders = [f.path for f in os.scandir(big_folder_path) if f.is_dir()]

# 给每个小文件夹编号
for idx, folder_path in enumerate(subfolders, start=1):
    new_folder_name = f"iden_{idx}"  # 定义新的文件夹名
    new_folder_path = os.path.join(big_folder_path, new_folder_name)
    os.rename(folder_path, new_folder_path)  # 重命名文件夹
'''
import os
import shutil

# 指定原始文件夹和新的文件夹
original_folder = "/home/wangm/log"
new_folder = "/opt/datasets/mnist/test"

# 确保新的文件夹存在，如果不存在则创建
os.makedirs(new_folder, exist_ok=True)

# 获取原始文件夹下所有以"train_"开头的文件
file_list = [file for file in os.listdir(original_folder) if file.startswith("test")]

# 遍历文件列表，重命名并移动文件
for old_filename in file_list:
    # 构造原始文件的完整路径
    old_filepath = os.path.join(original_folder, old_filename)

    # 构造新的文件名，去掉"train_"前缀
    new_filename = old_filename.replace("train_", "")

    # 构造新文件的完整路径
    new_filepath = os.path.join(new_folder, new_filename)

    # 重命名并移动文件
    shutil.move(old_filepath, new_filepath)

print("重命名并移动完成。")
