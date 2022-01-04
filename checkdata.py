from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy
from PIL import Image



images_origin = np.load('./datasets/cifar_train_image.npy')
images = np.load('./datasets/train_PGD-12_densenet_image.npy')
# labels = np.load('./cifar_color_label.npy')
# images_origin = images_origin / images_origin.sum(axis=1, keepdims=True)
# merge = (images_origin + images)/2.
# np.save('./cifar_distill_label.npy', merge)
# print(np.argmax(images_origin[0:10], axis=1))
# print(np.argmax(images[0:10], axis=1))
# print(np.argmax(merge[0:10], axis=1))
# print(images_origin[1])
# print(images[1])

showlist = [0,1,2,3,4]
for i in showlist:
    image_origin = Image.fromarray(images_origin[i])
    image = Image.fromarray(images[i])
    # if (image == image_origin):
    #     print("True")
    # else:
    #     print("False")
    # print(images[i][0][0])
    # #print(images_origin[i][0][0])
    # print("!!!")
    image_origin.save('./show/train'+str(i)+'.png')
    image.save('./show/t—PGD'+str(i)+'.png')
    # print(labels[i])
print(images.shape)
# print('origin:', images_origin[0][0][0])
# print('PGD:', images[0][0][0])
# np.save('data.npy', images_merge)
# np.save('label.npy', labels_merge)


#最优解1PGD-d+2PGD-d+3PGD-d+light+w10