#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:43:08 2019

@author: catle
"""

# Numpy
import numpy as np
from numpy import linalg as LA

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse
from kmeans_pytorch.kmeans import lloyd

# Parallel Module
class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
    
# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def main():    
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2000,
                                             shuffle=False, num_workers=2)
    
    dataset_size = 10000
    trainset_kmeans = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader_kmeans = torch.utils.data.DataLoader(trainset_kmeans, batch_size=10000, shuffle=False, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))))
#    trainloader_kmeans = torch.utils.data.DataLoader(trainset_kmeans, batch_size=30000, shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    iteration = 100

    for epoch in range(iteration):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            MSE_loss = nn.MSELoss()(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
        print('MSE LOSS: epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
                      .format(epoch + 1, i + 1, loss.item(), MSE_loss.item()))

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")
    
    
    
    
    
    # k-means process
    print('==> start K-means process..')    
    K = 150
    
    with torch.no_grad():
        for epoch in range(1):
            # Extract Feature
            autoencoder.eval()
            
            for i, (inputs, targets) in enumerate(trainloader_kmeans, 0):
                targets.cpu().detach().numpy()
                inputs = get_torch_vars(inputs)
                print(inputs.shape)
                features = autoencoder.encoder(inputs).cpu().detach().numpy()
                print(features.shape)
                
                features = np.reshape(features, (dataset_size, 768))
                print(features.shape)
            
#            for _, (inputs, targets) in enumerate(trainloader_kmeans):
#                inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
#                inputs = inputs.view(inputs.size(0), -1)
#                inputs = Variable(inputs).cuda()
#                print(inputs.shape)
#                features = autoencoder.encoder(inputs).cpu().detach().numpy()
#                print(features.shape)
                
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
    
    # Testing---------------------------------------------
    acc = 0
    
    def test(epoch, args):
        global best_acc
        global acc
        autoencoder.eval()
        correct = 0
        total = 0
    
        for i, (inputs, targets) in enumerate(testloader, 0):
            targets.cpu().detach().numpy()
            inputs = get_torch_vars(inputs)
            features = autoencoder.encoder(inputs).cpu().detach().numpy()
            features = np.reshape(features, (2000, 768))
                
                
#        for _, (inputs, targets) in enumerate(testloader):
#            inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
#            inputs = inputs.view(inputs.size(0), -1)
#            inputs = Variable(inputs).cuda()
#            features = autoencoder.encoder(inputs).cpu().detach().numpy()
            
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
#        if acc > best_acc:
#            print('Saving..')  
#            best_acc = acc
    
    test(epoch, args)


if __name__ == '__main__':
    main()
