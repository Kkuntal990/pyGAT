#!/usr/bin/python3

import numpy as np
from scipy.stats import poisson


n = int(input()) #number of features
#n = 15

idx = np.asarray([i for i in range(n)])
idx = np.reshape(idx,(-1,1))

#now to create features

features = np.array(poisson.rvs(mu=2,size=50))
features = np.reshape(features,(1,-1))
#print(features)
for _ in range(n-1):
	temp = np.array(poisson.rvs(mu=2,size=50))
	temp = np.reshape(temp,(1,-1))
	features = np.concatenate((features,temp), axis=0)

labels = np.random.randint(3,size=n)
labels = np.reshape(labels,(-1,1))

data = np.concatenate((idx,features,labels), axis=1)

m = np.matrix(data)

with open('./customData/data1.content','wb') as f:
	for line in m:
		np.savetxt(f,line, fmt='%2f')




