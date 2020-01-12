#!/usr/bin/python3

from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import torch.nn as nn
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import load_data, accuracy, load_data1
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--nodes', type=int, default=15,help='Number of nodes')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


adj, features, labels, idx_train = load_data1(args.nodes)

#print(adj)
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    

features, adj, labels = Variable(features), Variable(adj), Variable(labels)
#print(adj.shape)


def tsne_plot(data, num,epochs):

	tsne = TSNE(n_components =2, perplexity=7, n_iter=1000, method= 'exact',verbose=0)
	tsne_result = tsne.fit_transform(data)
	df = pd.DataFrame()
	df["one-tsne"] = tsne_result[:,0]
	df["two-tsne"] = tsne_result[:,1]
	df["y"] = labels
	ax = plt.subplot(2,3,num)
	ax.set_title('epoch {}/{}'.format((num-1)*150,epochs ))
	sns.scatterplot(x="one-tsne", y = "two-tsne",hue="y",palette=sns.color_palette("hls",3),data=df, legend="full",alpha=0.3,ax=ax)

	




def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.data.item()


t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
featuresStack = []
best_epoch = 0
for epoch in range(args.epochs):
	loss_values.append(train(epoch))
	torch.save(model.state_dict(), '{}.pkl'.format(epoch))
	if loss_values[-1] < best:
		best = loss_values[-1]
		best_epoch = epoch
		bad_counter = 0
	else:
		bad_counter+=1

	if bad_counter == args.patience:
		break

	if epoch%150==0:
		#print(epoch)
		#features = features.numpy()
		featuresStack.append(features)

	#features = features.cuda()

	files = glob.glob('*.pkl')
	for file in files:
		epoch_nb = int(file.split('.')[0])
		if epoch_nb < best_epoch:
			os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

plt.figure()
for i in range(len(featuresStack)):
	tsne_plot(featuresStack[i],i+1, args.epochs)

plt.title("best was {}, but this is 750th epoch".format(best_epoch))

plt.show()




# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
