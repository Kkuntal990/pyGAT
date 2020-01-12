#!/usr/bin/python3

import numpy as np
import random
n = int(input())
#now create edges
edges = np.array([[0,0]])
for i in range(n):
	for j in range(n):
		if i==j:
			continue
		temp = np.array([[i,j]])
		edges = np.concatenate((edges,temp), axis = 0)
#print(edges)

edges = edges[1:,:]

#total 210 edges possible, so selecting 100
indices = [i for i in range(n*(n-1))]
random.shuffle(indices)
#print(indices)
indices = indices[:int(n*(n-1)/2)]


selected = np.array([edges[i] for i in indices])
m = np.matrix(selected)
with open("./customData/data1.edge",'wb') as f:
	for line in m:
		#print(line)
		np.savetxt(f,line,delimiter=" ",fmt="%d")

