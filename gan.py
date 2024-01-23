import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)    # 用于储存生成结果
                                        # 文件夹名：images
                                        # exist_ok=True：如果images创建前已存在，不会抛出错误

## 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")                   # 轮次数
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")                          # 批次数
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")                             # 学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

## 包含图像形状信息的元组（通道数、高、宽），决定了全连接层中最后的输出神经元个数
img_shape = (opt.channels, opt.img_size, opt.img_size)

## 是否可用CUDA
cuda = True if torch.cuda.is_available() else False

## 定义生成器类
class Generator(nn.Module):     # 基于父类Module的子类
    def __init__(self):
        super(Generator, self).__init__()  # 父类初始化

        def block(in_feat, out_feat, normalize=True):           # 定义了一个块创建函数(输入特征数，输出特征数，是否使用批量归一化)
            layers = [nn.Linear(in_feat, out_feat)]             # 创建了一个包含各层的列表
            if normalize:                                       # 如果使用批量归一化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))    # 增加一个批量归一化层(上一层输出特征数，旧数据权重)
            layers.append(nn.LeakyReLU(0.2, inplace=True))      # LeakyReLU激活函数(负斜率参数，是否原地操作)
            return layers                                       # 返回块

        self.model = nn.Sequential(                             # 创建Sequential类对象：model模型
            *block(opt.latent_dim, 128, normalize=False),       # *将block()函数返回的层列表中的一个个元素单独解包出来最为独立参数给Sequential()
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),           # 全连接层
            nn.Tanh()                                           # Tanh激活函数，将输入值映射到[-1,1]范围内
        )

    def forward(self, z):                                       # 前向传播函数
        img = self.model(z)                                     # 将图像输入model模型
        img = img.view(img.size(0), *img_shape)                 # view()方法用于重塑张量形状，将(批次数,通道数*高度*宽度)——>(批次数，通道数，高度，宽度)
        return img                                              # 返回img对象

## 定义判别器类
class Discriminator(nn.Module):                      # 基于父类Module的子类
    def __init__(self):
        super(Discriminator, self).__init__()        # 父类初始化
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512), # 全连接层：输入图片总特征数 ——> 512
            nn.LeakyReLU(0.2, inplace=True),         # LeakyReLU损失函数：(负斜率参数，是否原地操作)
            nn.Linear(512, 256),                     # 全连接层：512 ——> 256
            nn.LeakyReLU(0.2, inplace=True),         # LeakyReLU激活函数：(负斜率参数，是否原地操作)
            nn.Linear(256, 1),                       # 全连接层: 256 ——> 1
            nn.Sigmoid(),                            # Sigmoid激活函数
        )

    def forward(self, img):                          # 前向传播函数
        img_flat = img.view(img.size(0), -1)         # 将img张量重塑为[批次数，通道数*高度*宽度]，使其适用于全连接层
        validity = self.model(img_flat)              # 将img_flat经过model模型
        return validity                              # 返回validity张量


## 创建BCE损失函数对象
adversarial_loss = torch.nn.BCELoss()

## 创建生成器和判别器对象
generator = Generator()
discriminator = Discriminator()

## 如果使用CUDA的话
if cuda:
    generator.cuda()            # 将生成器对象转移到GPU上
    discriminator.cuda()        # 将判别器对象转移到GPU上
    adversarial_loss.cuda()     # 将损失函数对象转移到GPU上

## 加载训练数据，生成一个批次一个批次的数据
os.makedirs("./data/mnist", exist_ok=True)  # 创建/data/mnist文件夹，先前存在不报错
dataloader = torch.utils.data.DataLoader(   # 定义数据加载器对象
    datasets.MNIST(                         # 从torchvision中获取MNIST数据集
        "./data/mnist",                     # 定义MNIST数据集的储存地点
        train=True,                         # 加载的是训练数据集
        download=True,                      # 如果MNIST数据集不在原定位置，自动下载
        transform=transforms.Compose(       # 数据转换
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                # Resize():将图片尺寸转换至命令行中定义的尺寸
                # ToTensor()：将图像转换为 PyTorch 张量
                # Normalize():标准化图像，使像素值范围在 [-1, 1] 之间,两个[0.5]分别是均差和标准差
        ),
    ),
    batch_size=opt.batch_size,              # 批次大小    
    shuffle=True,                           # 随机打乱数据
)

## 创建Adam优化器，负责梯度下降更新权重和偏置参数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # parameters():提取生成器对象中所有的可更新参数：权重参数、偏置参数
    # lr:学习率
    # betas：一阶矩、二阶矩中的衰减率
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor   # 创建张量对象   

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):                                        # enumerate：在遍历时同时获取每个元素的索引和值，分别给i和imgs
        # (imgs, _):获取每个元素包含的一对值，其中_表示忽略标签

        ## 定义标签
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)     # 创建“真实”标签
            # Variable():早期pytorch中创建张量的类，接受两个参数：具体张量、requires_grad（不需要对其进行梯度计算）
            # Tensor()函数：创建一个两个维度大小分别为imgs.size(0)、1的二位张量
            # fill_()方法：将张量内部的值都填充为1.0
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)      # 创建“假”标签
        real_imgs = Variable(imgs.type(Tensor))                                       # 确保imgs格式和Tensor类定义的张量格式相同

        # -----------------
        #  训练生成器
        # -----------------

        ## 前向传播
        optimizer_G.zero_grad()                                                       # 梯度清零
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) # 生成噪音特征值，作为种子输入生成器
            # Variable():创建张量对象
            # normal(）创建符合高斯分布的数组
            # 0，1：表示均差和标准差
            # (imgs.shape[0], opt.latent_dim):数组形状，(批次数，种子大小)
        gen_imgs = generator(z)                                                       # 生成器开始前向传播

        ## 损失函数
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)                     # 根据判别器给出的判别结果和“真实”标签计算损失

        ## 反向传播
        g_loss.backward()  

        ## 梯度下降更新参数                      
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------

        optimizer_D.zero_grad()   # 梯度清零

        ## 计算损失函数
        real_loss = adversarial_loss(discriminator(real_imgs), valid)           # 识别真实图片，判断其为真实图片的损失函数
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)    # 识别生成图片，判断其为假图片的损失函数
        d_loss = (real_loss + fake_loss) / 2                                    # 做平均

        ## 反向传播
        d_loss.backward()                                                       # 根据损失函数进行反向传播获取梯度值

        ## 梯度下降更新参数
        optimizer_D.step()                                                      # 利用梯度值更新参数

        # 打印输出
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i                              # 已完成的batch数
        if batches_done % opt.sample_interval == 0:                             # batch数到达了设定的存档点
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)  #保存当前生成的图像样本