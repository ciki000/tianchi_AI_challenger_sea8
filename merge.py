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


datasets = ['./datasets/cifar_train1', './datasets/cifar_train2', './datasets/cifar_train3', './datasets/cifar_train4', './datasets/train5_PGD_resnet']
# datasets = ['./datasets/cifar_train1', './datasets/train2_PGD-8_densenet', './datasets/train3_PGD-8_resnet', './datasets/train4_PGD-4_densenet', './datasets/train5_PGD-4_resnet']
images = []
labels = []
cnt = 0
for dataset in datasets:
    cur_images = np.load(dataset+'_image.npy')
    cur_labels = np.load(dataset+'_label.npy')
    for i in range(cur_images.shape[0]):
        cnt = cnt + 1
        images.append(cur_images[i])
        labels.append(cur_labels[i])

index = [x for x in range(cnt)]
random.shuffle(index)

random_images = []
random_labels = []
for i in range(cnt):
    random_images.append(images[index[i]])
    random_labels.append(labels[index[i]])

images_merge = np.array(random_images).astype(np.uint8)
labels_merge = np.array(random_labels)

print(images_merge.shape, labels_merge.shape)
np.save('data_7.npy', images_merge)
np.save('label_7.npy', labels_merge)