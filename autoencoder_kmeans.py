'''Train MNIST with PyTorch.'''
from __future__ import print_function 
import networkx as nx
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from numpy import linalg as LA
#from models import *
from kmeans_pytorch.kmeans import lloyd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='GWR CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size', default=60000, type=int, help='batch size')
parser.add_argument('--num-epoch', default=1, type=int, help='number of epochs')
parser.add_argument('--thresh-similar', default=0.15, type=float, help='threshold of similar measure')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dim', default=512, type=int, help='feature dimension')
parser.add_argument('--max-age', default=30, type=int, help='maximum age for edge')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_size = 60000 

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

trainset = MNIST(root='./data', train=True, download=True, transform=img_transform)
#trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))))
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

testset = MNIST(root='./data', train=False, download=True, transform=img_transform)
testloader = DataLoader(testset, batch_size=2000, shuffle=False)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# load pre-trained NN Model
#print('==> loading model..')
##net = VGG('VGG19')
#net = AE('AE')
#net = net.to(device)
#checkpoint = torch.load('./checkpoint/ckpt.t9')
#net.load_state_dict(checkpoint['net'])


# Autoencoder
if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 10 
batch_size = 128
learning_rate = 1e-3


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


dataset = MNIST('./data', train=True, transform=img_transform, download=True)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
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
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')
print('Saving..')
state = {
    'model': model.state_dict(),
    'epoch': epoch,
    }
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t9')



# main
print('==> start K-means process..')    
K = 100
with torch.no_grad():
    for epoch in range(args.num_epoch):
        # Extract Feature
        model.eval()
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
            inputs = inputs.view(inputs.size(0), -1)
            inputs = Variable(inputs).cuda()
            features = model.encoder(inputs).cpu().detach().numpy()
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
        inputs = inputs.view(inputs.size(0), -1)
        inputs = Variable(inputs).cuda()
        features = model.encoder(inputs).cpu().detach().numpy()
        
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
#        if not os.path.isdir('checkpoint_GWR'):
#            os.mkdir('checkpoint_GWR')
#        nx.write_gpickle(G,'./checkpoint_GWR/graph.gpickle')
#        np.save('./checkpoint_GWR/best_acc.npy', acc)
        best_acc = acc

test(epoch, args)

# Graph        
G = nx.Graph()

for i in range(dataset_size+K):
    if i < dataset_size:
        G.add_nodes_from([i],weight=features[i,:])
    else:
        G.add_nodes_from([i],weight=c[i-dataset_size,:])

#for i in range(dataset_size + K):
#    for j in range(dataset_size + K):
        
pos = nx.spring_layout(G)
color = ['gray', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'purple', 'pink', 'red','black'] 
node_list = []
node_color = []
node_size = 10

for node in G.nodes():
    if node < dataset_size:
        node_list.append(node)
        label = targets[node]
        node_color.append(color[label])
    else:
        node_list.append(node)
        label = 10
        node_color.append(color[label])
    
#normalize the size
max_size = np.max(node_size)
min_size = np.min(node_size)

plt.figure(figsize=(10,10)) 
nx.draw_networkx(G, pos=pos, nodelist=node_list, node_size=node_size, node_color=node_color, width=2, with_labels=False)            
plt.show()    

# Scatter Plot
plt.figure(figsize=(8,8))
plt.scatter(LA.norm(features, axis=1), LA.norm(features, np.inf, axis=1), c=cl, s= 30000 / len(features), cmap="tab10")
plt.scatter(LA.norm(c, axis=1), LA.norm(c, np.inf, axis=1), c='black', s=50, alpha=.8)
#plt.axis([0,8,0,4])
plt.show()

# Decode k values
x = Variable(torch.from_numpy(c)).cuda()
x = to_img(model.decoder(x).cpu().data)
save_image(x, './mlp_img/k_values.png')
            
