'''
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import ClassifierNet
import torch.nn.functional as F
from torchvision.utils import save_image

tarR_path = "/home/wangm/MIA/DCGAN-MIA/log/TarR/T-FaceScrub/50.pkl"
TarR = ClassifierNet.TClassifier(num_classes=530)
TarR.load_state_dict(torch.load(tarR_path))

log_path = "/home/wangm/MIA/DCGAN-MIA/log/target_iden_t_celeba_facescrub"
# 设置图片预处理
preprocess = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

picture_folder = "/opt/datasets/CelebA"

target_num=100
target_iden=torch.zeros(target_num)

for i in range(target_num):
    target_iden[i]=i+1
    output_category_path = os.path.join(log_path, f'{int(target_iden[i])}')
    os.makedirs(output_category_path, exist_ok=True)

total_iden_num=0

# 遍历大文件夹中的所有图片
for folder_name in os.listdir(picture_folder):
    folder_path = os.path.join(picture_folder, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path).convert('RGB')
                image_preprocess = preprocess(image)
                image_preprocess = image_preprocess.unsqueeze(0)  
     
                with torch.no_grad():
                    output = TarR.forward(image_preprocess)
                    after_softmax = F.softmax(output, dim=-1)
                    value,predict = torch.max(after_softmax, dim=-1)
                    
                if predict+1 in target_iden.tolist():
                    output_path=os.path.join(log_path,f'{int(predict)+1}')
                    save_image(image_preprocess,fp=f"{output_path}/{value}.png")
                    total_iden_num+=1
            
print(total_iden_num)
'''

## mnist_train mnist_test
# 导入包
import struct
import numpy as np
from PIL import Image
import torch 
from torchvision import datasets, transforms

class MnistParser:
   # 加载图像
   def load_image(self, file_path):

       # 读取二进制数据
       binary = open(file_path,'rb').read()

       # 读取头文件
       fmt_head = '>iiii'
       offset = 0

       # 读取头文件
       magic_number,images_number,rows_number,columns_number = struct.unpack_from(fmt_head,binary,offset)

       # 打印头文件信息
       print('图片数量:%d,图片行数:%d,图片列数:%d'%(images_number,rows_number,columns_number))

       # 处理数据
       image_size = rows_number * columns_number
       fmt_data = '>'+str(image_size)+'B'
       offset = offset + struct.calcsize(fmt_head)

       # 读取数据
       images = np.empty((images_number,rows_number,columns_number))
       for i in range(images_number):
           images[i] = np.array(struct.unpack_from(fmt_data, binary, offset)).reshape((rows_number, columns_number))
           offset = offset + struct.calcsize(fmt_data)
           # 每1万张打印一次信息
           if (i+1) % 10000 == 0:
               print('> 已读取:%d张图片'%(i+1))

       # 返回数据
       return images_number,rows_number,columns_number,images


   # 加载标签
   def load_labels(self, file_path):
       # 读取数据
       binary = open(file_path,'rb').read()

       # 读取头文件
       fmt_head = '>ii'
       offset = 0

       # 读取头文件
       magic_number,items_number = struct.unpack_from(fmt_head,binary,offset)

       # 打印头文件信息
       print('标签数:%d'%(items_number))

       # 处理数据
       fmt_data = '>B'
       offset = offset + struct.calcsize(fmt_head)

       # 读取数据
       labels = np.empty((items_number))
       for i in range(items_number):
           labels[i] = struct.unpack_from(fmt_data, binary, offset)[0]
           offset = offset + struct.calcsize(fmt_data)
           # 每1万张打印一次信息
           if (i+1)%10000 == 0:
               print('> 已读取:%d个标签'%(i+1))

       # 返回数据
       return items_number,labels


   # 图片可视化
   def visualaztion(self, images, labels, path):
       d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
       for i in range(images.__len__()):
            im = Image.fromarray(np.uint8(images[i]))
            im.save(path + "%d_%d.png"%(labels[i], d[labels[i]]))
            d[labels[i]] += 1
            # im.show()
            
            if (i+1)%10000 == 0:
               print('> 已保存:%d个图片'%(i+1))
               

# 保存为图片格式
def change_and_save():
    mnist =  MnistParser()

    # trainImageFile = '/home/wangm/log/MNIST_data/MNIST/raw/train-images-idx3-ubyte'
    # _, _, _, images = mnist.load_image(trainImageFile)
    # trainLabelFile = '/home/wangm/log/MNIST_data/MNIST/raw/train-labels-idx1-ubyte'
    # _, labels = mnist.load_labels(trainLabelFile)
    # mnist.visualaztion(images, labels, "/home/wangm/log/train")

    testImageFile = '/home/wangm/log/MNIST_data/MNIST/raw/t10k-images-idx3-ubyte'
    _, _, _, images = mnist.load_image(testImageFile)
    testLabelFile = '/home/wangm/log/MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte'
    _, labels = mnist.load_labels(testLabelFile)
    mnist.visualaztion(images, labels, "/home/wangm/log/test")


# 测试
if __name__ == '__main__':
    usps_train_dataset = datasets.USPS(root="/home/wangm/log", train=True, download=True)
    usps_test_dataset = datasets.USPS(root="/home/wangm/log", train=False, download=True)

    change_and_save()
