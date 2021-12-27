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


datasets = ['./datasets/cifar_test', './datasets/test_PGD-10_densenet', './datasets/test_2PGD-10_resnet']
images = []
labels = []
for dataset in datasets:
    cur_images = np.load(dataset+'_image.npy')
    cur_labels = np.load(dataset+'_label.npy')
    for i in range(cur_images.shape[0]):
        images.append(cur_images[i])
        labels.append(cur_labels[i])

images_merge = np.array(images).astype(np.uint8)
labels_merge = np.array(labels)

print(images_merge.shape, labels_merge.shape)
np.save('data.npy', images_merge)
np.save('label.npy', labels_merge)