#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:35:39 2019

@author: catle
"""

'''Train fashion-MNIST with PyTorch.'''
'''Pretrain decoder of autoencoder with fashion-MNIST'''
import os
import argparse

import numpy as np
from numpy import linalg as LA
from kmeans_pytorch.kmeans import lloyd

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image

class Flatten(torch.nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
device = 'cuda'

# Parser
parser = argparse.ArgumentParser(description='GWR CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=32, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=2000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=1, type=int, help='number of epochs')
parser.add_argument('--thresh-similar', default=0.15, type=float, help='threshold of similar measure')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dim', default=512, type=int, help='feature dimension')
parser.add_argument('--max-age', default=30, type=int, help='maximum age for edge')
args = parser.parse_args()

# Dataset
if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

def tensor_round(tensor):
    return torch.round(tensor)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

#trainset = MNIST(root='./data', train=True, download=True, transform=img_transform)
#trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=False)
#
#testset = MNIST(root='./data', train=False, download=True, transform=img_transform)
#testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
#classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

transform = transforms.ToTensor()

trainset = FashionMNIST('./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)

testset = FashionMNIST('./data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=True)

# Encoder of Autoencoder
class AEencoder(nn.Module):
    def __init__(self):
        super(AEencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                
                Flatten(),
                
                nn.Linear(28 * 28 * 64, 512),
                nn.ReLU(True),
                nn.Linear(512, 64),
                nn.ReLU(True))
        
        self.classifier = nn.Linear(64,10)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X

class AEdecoder(nn.Module):
    def __init__(self):
        super(AEdecoder, self).__init__()
        self.linear_decoder = nn.Sequential(
                nn.Linear(64, 512),
                nn.ReLU(True),
                nn.Linear(512, 28 * 28 * 64),
                nn.ReLU(True))
        
        self.conv_decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid())
        
    def forward(self,X):
        X = self.linear_decoder(X)
        X = X.view(X.size(0), 64, 28, 28)
        X = self.conv_decoder(X)
        return X

# Load CNN network
net = AEencoder()
net = net.to(device)
checkpoint = torch.load('./checkpoint/pretrained-encoder-fashion.t2')
net.load_state_dict(checkpoint['net'])
print(net)

#------------------------------------
# Modify pretrained CNN
#------------------------------------
# Freeze the parameters of encoder's layer 0, 2, 5, 7
# only linear layers
for param in net.parameters():
    param.requires_grad = False
#net.encoder[0].weight.requires_grad = False
#net.encoder[2].weight.requires_grad = False
#net.encoder[5].weight.requires_grad = False
#net.encoder[7].weight.requires_grad = False


# Remove classifier (last) layer
model = nn.Sequential(*list(net.children())[:-1])

# Add new layers (decoder layers)
decoder_net = AEdecoder()
decoder_net = decoder_net.to(device)
model = torch.nn.Sequential(model, decoder_net)

# Train the Autoencoder
num_epochs = 10
batch_size = 128
learning_rate = 1e-3

#dataset = FashionMNIST('./data', download=True, train=True, transform=transform)
#dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = model.to(device)
print(model)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in trainloader:
        img, _ = data
        img = img.to(device)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item(), MSE_loss.item()))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)

print('Saving..')
state = {
    'model': model.state_dict(),
    'epoch': epoch,
    }
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/AE.t1')










#---------------------------------------------------
# Quantizer (k-means)
#---------------------------------------------------
# load entire dataset for k-means
dataset_size = 60000

print('==> start K-means process..')    
K = 50
with torch.no_grad():
    for epoch in range(args.num_epoch):
        # Extract Feature
        model.eval()
        features = []
        targets = []
        for _, (inputs, target) in enumerate(trainloader):
            inputs, target = inputs.to(device), target.cpu().detach().numpy()
            feature = model[0](inputs).cpu().detach().numpy()
            features.append(feature)
            targets.append(target)

        features = np.concatenate(features,axis=0)
        targets = np.concatenate(targets,axis=0)
        print(features.shape)
        
        cl, c = lloyd(features, K, device=0, tol=1e-4)
        print('next batch...')

k_label = []
hit_map = np.column_stack((cl, targets))


for i in range(K):
    temp = []
    for j in range(dataset_size):
        if hit_map[j,0] == i:
            temp.append(hit_map[j,1])
        
    hist = np.histogram(temp,bins=10,range=(0,9))[0]
    index = [idx for idx, cnt in enumerate(hist) if cnt == max(hist)]
    k_label.append(index[0])

print('finishing training k-means...')

# Testing
best_acc = 0
acc = 0
def test(epoch, args):
    global best_acc
    global acc
    model.eval()
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
        features = model[0](inputs).cpu().detach().numpy()
        
        dist = np.zeros((inputs.size(0), K))
        for node in range(K):
            # L2 distance measure
            dist[:,node] = LA.norm(c[node] - features, axis=1)
        closest_id = np.argpartition(dist, 0, axis=1)   
        
        for idx in range(inputs.size(0)):
            if k_label[closest_id[idx,0]] == targets[idx]:
                correct += 1
        total += len(targets)

    if epoch % 1 == 0:   
        print('--Test Acc-- (epoch=%d): %.3f%% (%d/%d)' % (epoch, 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')  
        best_acc = acc

test(epoch, args)

#-------------------------------------
# Test decoder
#-------------------------------------
# Decode k values
x = Variable(torch.from_numpy(c)).cuda()
x = to_img(model[1](x).cpu().data)
save_image(x, './mlp_img/k_values.png')

#-------------------------------------------------   
# Find images that mapped to cluster point
#-------------------------------------------------
# Find images mapped to each cluster
for j in range(K):
    cluster_i = np.zeros(64).astype(np.float32)
    for i in range(dataset_size):
        if cl[i] == j:
            cluster_i = np.c_[cluster_i, features[i]]
            
    cluster_i = np.transpose(cluster_i)
    cluster_i = np.delete(cluster_i, (0), axis=0)
    x_cluster_i = Variable(torch.from_numpy(cluster_i)).cuda()
    x_cluster_i = to_img(model[1](x_cluster_i).cpu().data)
    save_image(x_cluster_i, './mlp_img/all_cluster/cluster_{}.png'.format(j))


# Find images mapped to the 0th cluster
cluster0 = np.zeros(64).astype(np.float32)
for i in range(dataset_size):
    if cl[i] == 1:
        cluster0 = np.c_[cluster0, features[i]]
        
cluster0 = np.transpose(cluster0)
cluster0 = np.delete(cluster0, (0), axis=0)
x_cluster0 = Variable(torch.from_numpy(cluster0)).cuda()
x_cluster0 = to_img(model[1](x_cluster0).cpu().data)
save_image(x_cluster0, './mlp_img/cluster0.png')

#-------------------------------------------------
# Convex Hull for generativity
#-------------------------------------------------
convex = cluster0[0]
temp1 = cluster0[0]
temp2 = cluster0[1]
temp3 = cluster0[10]
alpha = np.linspace(0,1,8)

for i in alpha:
    for j in (1-alpha):
        temp = i*temp3 + j*temp2 + (1-i-j)*temp1
        convex = np.c_[convex,temp]

convex = np.transpose(convex)
convex = np.delete(convex, (0), axis=0)
x_convex = Variable(torch.from_numpy(convex)).cuda()
x_convex = to_img(model[1](x_convex).cpu().data)
save_image(x_convex, './mlp_img/convex.png')


convex_sample = temp1
convex_sample = np.c_[convex_sample,temp2]
convex_sample = np.c_[convex_sample,temp3]
convex_sample = np.transpose(convex_sample)
convex_sample = Variable(torch.from_numpy(convex_sample)).cuda()
convex_sample = to_img(model[1](convex_sample).cpu().data)
save_image(convex_sample, './mlp_img/convex_representatives.png')


