from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy

from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.cw import CarliniWagner

import torchattacks
from torchattacks import CW
import cv2

def lightchange(image):
    
    light = random.uniform(0.8, 0.9)
    image  = np.power(image, light)
 
    return image

image = np.load('./datasets/cifar_test_image.npy')
label = np.load('./datasets/cifar_test_label.npy')
# light
# for i in range(image.shape[0]):
#     image[i] = lightchange(image[i])

# blur
for i in range(image.shape[0]):
     image[i] = cv2.GaussianBlur(image[i], (3,3), 0)

image = np.round(image).astype(np.uint8)
#print(image.shape)
np.save('./datasets/test_blur_image.npy', image)
np.save('./datasets/test_blur_label.npy', label)