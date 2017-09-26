import numpy as np
import csv
import math
import operator
import random

def relu(Z):
	return(np.maximum(Z,0),Z)


def initialize_parameters_deep(layer_dims):
	parameters={}
	L=len(layer_dims)
	for l in range(1,L):
		parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters["b"+str(l)]=np.zeros((layer_dims[l],1))

		assert(parameters["W"+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
		assert(parameters["b"+str(l)].shape==(layer_dims[l],1))

	return parameters

def linear_forward(A,W,b):
	Z=np.dot(W,A)+b	
	assert(Z.shape==(W.shape[0],A.shape[1]))
	cache=(A,W,b)
	return Z,cache

def linear_actiavtion_forward(A_prev,W,b,activation):
	if activation=="relu":
		Z,linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=relu(Z)
	if activation=="softmax":

	assert(A.shape==(W.shape[0],A_prev.shape[1]))
	cache=(linear_cache,actiavtion_cache)

	return A,cache

def L_model_forward(X,parameters):
	caches=[]
	A=X
	L=len(parameters)
	for l in range(1,L):
		A_prev=A
		A,cache=linear_actiavtion_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="relu")
		caches.append(cache)
	AL,cache=linear_actiavtion_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="softmax")
			



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

