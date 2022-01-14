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
    def __init__(self, images, labels, transform):
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


use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def test(resnet, densenet, images, labels):
    resnet.cuda()
    densenet.cuda()
    resnet.eval()
    densenet.eval()

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = MyDataset(images, labels, transform_test)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)

    resnet_accs = AverageMeter()
    densenet_accs = AverageMeter()
    for (input_, soft_label) in tqdm(testloader):
        input_, soft_label = input_.cuda(), soft_label.cuda()
        target = soft_label.argmax(dim=1)

        resnet_output = resnet(input_)
        densenet_output = densenet(input_)

        resnet_acc = accuracy(resnet_output, target)
        densenet_acc = accuracy(densenet_output, target)
            
        resnet_accs.update(resnet_acc[0].item(), input_.size(0))
        densenet_accs.update(densenet_acc[0].item(), input_.size(0))

    print('Test acc:')
    print(resnet_accs.avg, densenet_accs.avg)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

# def FGSM(model, x, label, eps=0.001):
#     x_new = x 
#     x_new = Variable(x_new, requires_grad=True)
    
#     y_pred = model(x_new)
#     loss = cross_entropy(y_pred, label)

#     model.zero_grad()
#     loss.backward()
#     grad = x_new.grad.cpu().detach().numpy()
#     grad = np.sign(grad)
#     pertubation = grad * eps
#     adv_x = x.cpu().detach().numpy() + pertubation
#     #adv_x = np.clip(adv_x, clip_min, clip_max)

#     x_adv = torch.from_numpy(adv_x).cuda()
#     return x_adv

# def PGD(models, x, label, it=10, eps=0.001):
    
#     for i in range(it):
#         for model in models:
#             x = FGSM(model, x, label, eps)
#     return x
class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def attack(resnet, densenet, images, labels, id):
    # resnet.eval()
    # densenet.eval()
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
    ])
    testset = MyDataset(images, labels, transform_test)
    testloader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)

    # inputs_adv = []
    # for (input_, soft_label) in tqdm(testloader):
    #     input_, soft_label = input_.cuda(), soft_label.cuda()

    #     models = [resnet, densenet]
    #     x = Variable(input_)
    #     x = PGD(models, x, soft_label, it=it)

    #     inv_normalize = transforms.Normalize((-2.4290657439446366, -2.418254764292879, -2.2213930348258706), (4.9431537320810675, 5.015045135406218, 4.975124378109452))
    #     inputs_adv.append(np.clip(inv_normalize(x.squeeze()).cpu().numpy().transpose((1,2,0))*256, 0, 256))

    # images_adv = np.array(inputs_adv).astype(np.uint8)
    # np.save('cifar_adv'+str(id+1)+'_image.npy', images_adv)
    # np.save('cifar_adv'+str(id+1)+'_label.npy', labels)
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    norm_resnet = nn.Sequential(
        norm_layer,
        resnet
    ).cuda()
    norm_resnet.eval()

    norm_densenet = nn.Sequential(
        norm_layer,
        densenet
    ).cuda()
    norm_densenet.eval()

    inputs_adv = []
    for (input_, soft_label) in tqdm(testloader):
        input_, soft_label = input_.cuda(), soft_label.cuda()

        x = Variable(input_)
        attack_labels = torch.topk(soft_label, 1)[1].squeeze(1)
        atk_densenet = PGD(norm_densenet, eps=10/255, alpha=2/255, steps=40, random_start=False)
        adv_images = atk_densenet(x, attack_labels)

        for i in range(x.shape[0]):
            inputs_adv.append(np.clip(x[i].squeeze().cpu().detach().numpy().transpose((1,2,0)), 0, 1)*255)

    images_adv = np.array(inputs_adv).astype(np.uint8)
    np.save('cifar_'+str(id+1)+'PGD-10_image.npy', images_adv)
    np.save('cifar_'+str(id+1)+'PGD-10_label.npy', labels)
    return images_adv, labels

def merge(imagelist, labellist):
    images = []
    labels = []
    for i in range(len(imagelist)):
        cur_images = imagelist[i]
        cur_labels = labellist[i]
        for i in range(cur_images.shape[0]):
            images.append(cur_images[i])
            labels.append(cur_labels[i])

    images_merge = np.array(images).astype(np.uint8)
    labels_merge = np.array(labels)

    return images_merge, labels_merge

def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def defense(resnet, densenet, images, labels):
    for arch in ['resnet50', 'densenet121']:
        if arch == 'resnet50':
            args = args_resnet
            model = resnet
        else:
            args = args_densenet
            model = densenet
        
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(images, labels, transform_train)
        trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        # Model

        model.eval()
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        epochs = 200
        for epoch in tqdm(range(epochs)):

            train_loss, train_acc = train(trainloader, model, optimizer)
            # print(args)
            # print('acc: {}'.format(train_acc))

            # save model
            best_acc = max(train_acc, best_acc)
            # save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'acc': train_acc,
            #         'best_acc': best_acc,
            #         'optimizer' : optimizer.state_dict(),
            #     }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()

        print('Train acc:')
        print(best_acc)

def main():
    resnet = load_model('resnet50')
    resnet.cuda()
    #resnet.load_state_dict(torch.load('./checkpoints/resnet_base.pth')['state_dict'])
    densenet = load_model('densenet121')
    densenet.cuda()
    #densenet.load_state_dict(torch.load('./checkpoints/densenet_base.pth')['state_dict'])
    
    images_base = np.load('./datasets/cifar_test_image.npy')
    labels_base = np.load('./datasets/cifar_test_label.npy')
    images = images_base
    labels = labels_base
    
    
    for i in range(5):
        print("-----iter:", i, '-----')
        defense(resnet, densenet, images, labels)
        test(resnet, densenet, images, labels)
        images_adv, labels_adv = attack(resnet, densenet, images_base, labels_base, i)
        images, labels = merge([images, images_adv], [labels, labels_adv])
        test(resnet, densenet, images, labels)

if __name__ == '__main__':
    main()
