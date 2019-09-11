#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:32:36 2019

@author: catle
"""
import os
import argparse
import matplotlib as plt

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
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

# Flatten layer in module
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

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")

def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), ])
    
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#transform = transforms.ToTensor()
#
#trainset = FashionMNIST('./data', download=True, train=True, transform=transform)
#trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
#
#testset = FashionMNIST('./data', download=True, train=False, transform=transform)
#testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=True)

# Encoder of Autoencoder
class AEencoder(nn.Module):
    def __init__(self):
        super(AEencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(True),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(True),
            
            Flatten(),
                
            nn.Linear(2 * 2 * 96, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.ReLU(True))
        
        self.classifier = nn.Linear(64,10)
        
    def forward(self,X):
        X = self.encoder(X)
        X = self.classifier(X)
        return X
 
cnn = AEencoder().cuda()
print(cnn)

# Train the encoder
def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 50
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = error(output, targets)
            loss.backward()
            optimizer.step()
            
            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == targets).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} ({:.0f}%)\t Accuracy:{:.3f}%'.format(
                    epoch, 100.*batch_idx / len(train_loader), 
                    float(correct*100) / float(args.batch_size_train*(batch_idx+1))))

fit(cnn, trainloader)

# Test the encoder on test data
best_acc = 0
def evaluate(model, test_loader):
    global best_acc
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = test_imgs.to(device)
        test_labels = test_labels.to(device)
        
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( 
            float(correct*100) / float((len(test_loader)*args.batch_size_test))))
    
    # Save the pretrained network
    if correct > best_acc:
        print('Saving..')
        state = {
            'net': cnn.state_dict(),
            'acc': correct,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/pretrained-encoder-cifar10.t1')
        best_acc = correct

evaluate(cnn, testloader)










#---------------------------------------------------
# Quantizer (k-means)
#---------------------------------------------------
# load entire dataset for k-means
dataset_size = 50000

print('==> start K-means process..')    
K = 50
with torch.no_grad():
    for epoch in range(args.num_epoch):
        # Extract Feature
        cnn.eval()
        features = []
        targets = []
        for batch_idx, (inputs, target) in enumerate(trainloader):
            inputs, target = inputs.to(device), target.cpu().detach().numpy()
            feature = cnn.encoder(inputs).cpu().detach().numpy()
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
    cnn.eval()
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
        features = cnn.encoder(inputs).cpu().detach().numpy()
        
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
