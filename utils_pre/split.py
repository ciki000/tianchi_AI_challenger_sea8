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


images = np.load('./datasets/cifar_train_image.npy')
labels = np.load('./datasets/cifar_train_label.npy')
classes = 10
split_num = 5

split_classes = {}
for i in range(classes):
    split_classes[i] = []

split_datasets = {}
for i in range(split_num):
    split_datasets[i] = []

for i in range(images.shape[0]):
    label = np.argmax(labels[i])
    split_classes[label].append(i)

for i in range(classes):
    random.shuffle(split_classes[i])

dataset_size = images.shape[0]//split_num//classes
#print(dataset_size)
for i in range(split_num):
    for j in range(classes):
        for k in range(i*dataset_size, (i+1)*dataset_size):
            split_datasets[i].append(split_classes[j][k])
    random.shuffle(split_datasets[i])

for i in range(split_num):
    cur_images = []
    cur_labels = []
    for j in range(len(split_datasets[i])):
        cur_images.append(images[split_datasets[i][j]])
        cur_labels.append(labels[split_datasets[i][j]])
    
    images_split = np.array(cur_images).astype(np.uint8)
    labels_split = np.array(cur_labels)
    
    print(images_split.shape, labels_split.shape)
    np.save('./datasets/cifar_train'+str(i+1)+'_image.npy', images_split)
    np.save('./datasets/cifar_train'+str(i+1)+'_label.npy', labels_split)