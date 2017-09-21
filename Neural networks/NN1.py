import numpy as np
import csv
import math
import operator
import random

def sigmoid(z):
	s=1/(1+np.exp(-z))
	return s

def layer_sizes(X,Y):
	n_x	= X.shape[0]
	n_y = Y.shape[0]
	n_h = 4
	return (n_x,n_h,n_y)


def initialize_paramteres(n_x,n_h,n_y):
	W1=np.random.randn(n_h,n_x)*0.01
	b1=np.zeros((n_h,1))
	W2=np.random.randn(n_y,n_h)*0.01
	b2=np.zeros((n_y,1))

	assert(W1.shape==(n_h,n_x))
	assert(b1.shape==(n_h,1))
	assert(W2.shape==(n_y,n_h))
	assert(b2.shape==(n_y,1))

	parameters={"W1":W1,
				"b1":b1,
				"W2":W2,
				"b2":b2}
	return parameters
				


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

print("shape of x ="+str(X.shape))
print("shape of y ="+str(Y.shape))

n_x,n_h,n_y=layer_sizes(X,Y)
print("size of imput layer is n_x ="+str(n_x))
print("size of hidden layer is n_xh ="+str(n_h))
print("size of output layer is n_y ="+str(n_y))

parameters= initialize_paramteres(n_x,n_h,n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
