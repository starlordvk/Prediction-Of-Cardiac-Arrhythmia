import numpy as np
import csv
import math
import operator
import random

def initialize_parameters_deep(layer_dims):
	parameters={}
	L=len(layer_dims)
	for l in range(1,L):
		parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters["b"+str(l)]=np.zeros((layer_dims[l],1))

		assert(parameters["W"+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
		assert(parameters["b"+str(l)].shape==(layer_dims[l],1))

	return parameters

def linear_forward():
	

reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=np.array(X)
X=X.astype(np.float)
X=np.transpose(X)

reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=np.array(Y)
Y=Y.astype(np.int)

Y=np.transpose(Y)

#one hot encoding for multiclass
one_hot_encoded=list()
for value in range (0,Y.shape[1]):
	out=list()
	out=[0 for i in range(13)]
	out[Y[0][value]-1]=1
	one_hot_encoded.append(out)


Y=one_hot_encoded
Y=np.array(Y)
Y=np.transpose(Y)
print(Y)

print("shape of x ="+str(X.shape))
print("shape of y ="+str(Y.shape))		

