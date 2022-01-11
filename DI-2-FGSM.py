from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

import torchattacks
from torchattacks import CW, PGD

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./datasets/test_light_image.npy')
        labels = np.load('./datasets/test_light_label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)


class MIDIFGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(MIFGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        alpha = self.config['eps'] / self.config['attack_steps'] 
        decay = 1.0
        x_adv = x.detach().clone()
        momentum = torch.zeros_like(x_adv, device=x.device)
        if self.config['random_init'] :
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
            x_adv = torch.clamp(x_adv,*self.clamp)


        for step in range(self.config['attack_steps']):
            x.requires_grad=True
            logit = self.model(x)
            if self.target is None:
                cost = -F.cross_entropy(logit, y)
            else:
                cost = F.cross_entropy(logit, target)
            grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum*decay
            momentum = grad
            x_adv = x - alpha*grad.sign()
            a = torch.clamp(x - self.config['eps'], min=0)
            b = (x_adv >= a).float()*x_adv + (a > x_adv).float()*a
            c = (b > x + self.config['eps']).float() * (x + self.config['eps']) + (
                x + self.config['eps'] >= b
            ).float() * b
            x = torch.clamp(c, max=1).detach()
        x_adv = torch.clamp(x, *self.clamp)
        return x_adv

def attack(models, x, y, iter=10, eps=0.001):

    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    norm_resnet = nn.Sequential(
        norm_layer,
        models[0]
    ).cuda()
    norm_resnet.eval()

    norm_densenet = nn.Sequential(
        norm_layer,
        models[1]
    ).cuda()
    norm_densenet.eval()

    labels = torch.topk(y, 1)[1].squeeze(1)
    atk = PGD(norm_resnet, eps=4/255, alpha=2/255, steps=40, random_start=False)
    adv_images = atk_resnet(x, labels)
    return adv_images

use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Data
transform_test = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = MyDataset(transform=transform_test)
testloader = data.DataLoader(testset, batch_size=128, shuffle=False)

# Model
resnet = load_model('resnet50').cuda()
resnet.load_state_dict(torch.load('./checkpoints/resnet_test_drrlw.pth')['state_dict'])
resnet.eval()
densenet = load_model('densenet121').cuda()
densenet.load_state_dict(torch.load('./checkpoints/densenet_test_drrlw.pth')['state_dict'])
densenet.eval()

resnet_accs = AverageMeter()
densenet_accs = AverageMeter()
inputs_adv = []
labels = []
cnt = 0
for (input_, soft_label) in tqdm(testloader):
    input_, soft_label = input_.cuda(), soft_label.cuda()

    models = [resnet, densenet]
    x = Variable(input_)
    x = attack(models, x, soft_label)

    inv_normalize = transforms.Normalize((-2.4290657439446366, -2.418254764292879, -2.2213930348258706), (4.9431537320810675, 5.015045135406218, 4.975124378109452))
    for i in range(x.shape[0]):
        #inputs_adv.append(np.clip(inv_normalize(x[i].squeeze()).cpu().detach().numpy().transpose((1,2,0)), 0, 1)*255)
        inputs_adv.append(np.clip(x[i].squeeze().cpu().detach().numpy().transpose((1,2,0)), 0, 1)*255)
        labels.append(soft_label[i].squeeze().cpu().numpy())

    # cnt = cnt + 1
    # if (cnt >= 100):
    #     break

#images_adv = np.array(inputs_adv).astype(np.uint8)
images_adv = np.round(np.array(inputs_adv)).astype(np.uint8)
labels_adv = np.array(labels)

np.save('./datasets/test_light_r4_image.npy', images_adv)
np.save('./datasets/test_light_r4_label.npy', labels_adv)