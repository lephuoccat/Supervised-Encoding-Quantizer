'''Train MNIST with PyTorch.'''
from __future__ import print_function 
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

#from kmeans_pytorch.kmeans import lloyd
from gmm import GaussianMixture

from math import sqrt
import numpy as np
import os
import argparse
from numpy import linalg as LA

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

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

trainset = MNIST(root='./data', train=True, download=True, transform=img_transform)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

testset = MNIST(root='./data', train=False, download=True, transform=img_transform)
testloader = DataLoader(testset, batch_size=2000, shuffle=False)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Autoencoder
if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 5
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


#------------------
# main
#------------------
print('==> start GMM process..')    
K = 10
with torch.no_grad():
    for epoch in range(args.num_epoch):
        # Extract Feature
        model.eval()
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.cpu().detach().numpy()
            inputs = inputs.view(inputs.size(0), -1)
            inputs = Variable(inputs).cuda()
            features = model.encoder(inputs)
        
        gmm_model = GaussianMixture(10, 64).cuda()
        gmm_model.fit(features)
        cluster = gmm_model.predict(features)
        print(cluster)
        print(cluster.shape)
        print('next batch...')
        print(cluster[0:10])
        print(cluster[100:110])

print(cluster)
print(cluster.shape)
        
k_label = []
hit_map = np.column_stack((cluster, targets))

for i in range(K):
    temp = []
    for j in range(dataset_size):
        if hit_map[j,0] == i:
            temp.append(hit_map[j,1])
        
    hist = np.histogram(temp,bins=10,range=(0,9))[0]
    index = [idx for idx, cnt in enumerate(hist) if cnt == max(hist)]
    k_label.append(index[0])

print('finishing training GMM...')


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
        features = model.encoder(inputs)
        
        pred_label = gmm_model.predict(features)
        
        for idx in range(inputs.size(0)):
            if pred_label[idx] == targets[idx]:
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

print(acc)
print(best_acc)

## Graph        
#G = nx.Graph()
#
#for i in range(dataset_size+K):
#    if i < dataset_size:
#        G.add_nodes_from([i],weight=features[i,:])
#    else:
#        G.add_nodes_from([i],weight=c[i-dataset_size,:])
#
##for i in range(dataset_size + K):
##    for j in range(dataset_size + K):
#        
#pos = nx.spring_layout(G)
#color = ['gray', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'purple', 'pink', 'red','black'] 
#node_list = []
#node_color = []
#node_size = 10
#
#for node in G.nodes():
#    if node < dataset_size:
#        node_list.append(node)
#        label = targets[node]
#        node_color.append(color[label])
#    else:
#        node_list.append(node)
#        label = 10
#        node_color.append(color[label])
#    
##normalize the size
#max_size = np.max(node_size)
#min_size = np.min(node_size)
#
#plt.figure(figsize=(10,10)) 
#nx.draw_networkx(G, pos=pos, nodelist=node_list, node_size=node_size, node_color=node_color, width=2, with_labels=False)            
#plt.show()    
#
## Scatter Plot
#plt.figure(figsize=(8,8))
#plt.scatter(LA.norm(features, axis=1), LA.norm(features, np.inf, axis=1), c=cl, s= 30000 / len(features), cmap="tab10")
#plt.scatter(LA.norm(c, axis=1), LA.norm(c, np.inf, axis=1), c='black', s=50, alpha=.8)
##plt.axis([0,8,0,4])
#plt.show()
#
## Decode k values
#x = Variable(torch.from_numpy(c)).cuda()
#x = to_img(model.decoder(x).cpu().data)
#save_image(x, './mlp_img/k_values.png')
#
## random datapoint
#for i in range(K):
#    noise = np.random.normal(0, 5, [100,64]).astype(np.float32)
#    A = c[i] + noise
#    x_rd = Variable(torch.from_numpy(A)).cuda()
#    x_rd = to_img(model.decoder(x_rd).cpu().data)
#    save_image(x_rd, './mlp_img/k_rd_{}.png'.format(i))
#    
##-------------------------------------------------   
## Find images that mapped to cluster point
##-------------------------------------------------
##Find images mapped to the 0th cluster
#cluster0 = np.zeros(64).astype(np.float32)
#for i in range(dataset_size):
#    if cl[i] == 0:
#        cluster0 = np.c_[cluster0, features[i]]
#        
#cluster0 = np.transpose(cluster0)
#cluster0 = np.delete(cluster0, (0), axis=0)
#x_cluster0 = Variable(torch.from_numpy(cluster0)).cuda()
#x_cluster0 = to_img(model.decoder(x_cluster0).cpu().data)
#save_image(x_cluster0, './mlp_img/x_cluster0.png')
#
##Find images mapped to the 4th cluster
#cluster4 = np.zeros(64).astype(np.float32)
#for i in range(dataset_size):
#    if cl[i] == 4:
#        cluster4 = np.c_[cluster4, features[i]]
#        
#cluster4 = np.transpose(cluster4)
#cluster4 = np.delete(cluster4, (0), axis=0)
#x_cluster4 = Variable(torch.from_numpy(cluster4)).cuda()
#x_cluster4 = to_img(model.decoder(x_cluster4).cpu().data)
#save_image(x_cluster4, './mlp_img/x_cluster4.png')
#
##Find images mapped to the 98th cluster
#cluster98 = np.zeros(64).astype(np.float32)
#for i in range(dataset_size):
#    if cl[i] == 4:
#        cluster98 = np.c_[cluster98, features[i]]
#        
#cluster98 = np.transpose(cluster98)
#cluster98 = np.delete(cluster98, (0), axis=0)
#x_cluster98 = Variable(torch.from_numpy(cluster98)).cuda()
#x_cluster98 = to_img(model.decoder(x_cluster98).cpu().data)
#save_image(x_cluster98, './mlp_img/x_cluster98.png')
##-------------------------------------------------
## Convex Hull for generativity
##-------------------------------------------------
#convex_2 = cluster0[0]
#temp1 = cluster0[0]
#temp2 = cluster4[0]
#alpha = np.linspace(0.1,1,7)
#
#for i in alpha:
#    temp = i*temp2 + (1-i)*temp1
#    convex_2 = np.c_[convex_2,temp]
#
#convex_2 = np.transpose(convex_2)
#x_convex_2 = Variable(torch.from_numpy(convex_2)).cuda()
#x_convex_2 = to_img(model.decoder(x_convex_2).cpu().data)
#save_image(x_convex_2, './mlp_img/x_convex_2.png')
#
#
#
## 3 point convex hull
#convex_2 = cluster0[0]
#temp1 = cluster0[0]
#temp2 = cluster4[0]
#temp3 = cluster98[0]
#alpha = np.linspace(0,1,8)
#
#for i in alpha:
#    for j in alpha:
#        temp = j*temp3 + i*temp2 + (1-i-j)*temp1
#        convex_2 = np.c_[convex_2,temp]
#
#convex_2 = np.transpose(convex_2)
#convex_2 = np.delete(convex_2, (0), axis=0)
#x_convex_2 = Variable(torch.from_numpy(convex_2)).cuda()
#x_convex_2 = to_img(model.decoder(x_convex_2).cpu().data)
#save_image(x_convex_2, './mlp_img/x_convex_3.png')







