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



images = np.load('./data.npy')

print(images.shape)
h = images.shape[1]
w = images.shape[2]
print(h, w)
for i in range(images.shape[0]):
    images[i, 0, :, :] = 0
    images[i, h-1, :, :] = 0
    images[i, :, 0, :] = 0
    images[i, :, w-1, :] = 0
print(images.shape)
np.save('data_box.npy', images)
