import numpy as np
import csv
import math
import operator
import random
from scipy.special import expit

#sigmoid function definition
def sigmoid(z):
	s=1.0/(1+np.exp(-z))
	return s

#softmax function for multiclass classification
def softmax(z):
	s=np.exp(z)/np.sum(np.exp(z),axis=0)	
	return s

#initializing the layer sizes, assuming the hidden layer has 4 units
def layer_sizes(X,Y):
	n_x	= X.shape[0]
	n_y = Y.shape[0]
	n_h = 10
	return (n_x,n_h,n_y)

#initializing weights and biases for layer 1 and layer 2
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

#forward propagation to compute activation values. 13 is the number of classes 
def forward_propagation	(X,parameters):
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]
	
	Z1=np.dot(W1,X)+b1
	A1=np.tanh(Z1)
	Z2=np.dot(W2,A1)+b2
	A2=softmax(Z2)

	assert(A2.shape==(13,X.shape[1]))

	cache={"Z1":Z1,
			"A1":A1,
			"Z2":Z2,
			"A2":A2}
	return A2,cache

#computes cost based on softmax function
def compute_cost(A2,Y,parameters):
	m=Y.shape[1]
	logprobs=(np.multiply(np.log(A2),Y))
	cost=-np.sum(logprobs)/m
	cost=float(cost)
	assert(isinstance(cost,float))

	return cost
						
#computes gradients for learning
def backward_propagation(parameters,cache,X,Y):
	m=X.shape[1]
	W1=parameters["W1"]
	W2=parameters["W2"]

	A1=cache["A1"]
	A2=cache["A2"]

	dZ2=A2-Y
	dW2=np.dot(dZ2,A1.T)/m
	db2=np.sum(dZ2,axis=1,keepdims=True)/m
	dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
	dW1=np.dot(dZ1,X.T)/m
	db1=np.sum(dZ1,axis=1,keepdims=True)/m

	grads={"dW1":dW1,
		   "db1":db1,
		   "dW2":dW2,
		   "db2":db2}	
	return grads

#update parameters based on learning rate and gradients dW and db
def update_paramters(parameters,grads,learning_rate):
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]

	dW1=grads["dW1"]
	db1=grads["db1"]
	dW2=grads["dW2"]
	db2=grads["db2"]		
		 
	W1=W1-learning_rate*dW1
	b1=b1-learning_rate*db1
	W2=W2-learning_rate*dW2
	b2=b2-learning_rate*db2	   	

	parameters={"W1":W1,
				"b1":b1,
				"W2":W2,
				"b2":b2}
	return parameters

#model for NN
def nn_model(X,Y,n_h,num_iterations=1000, print_cost=False):
	n_x,n_h,n_y=layer_sizes(X,Y)
	print("size of imput layer is n_x ="+str(n_x))
	print("size of hidden layer is n_xh ="+str(n_h))
	print("size of output layer is n_y ="+str(n_y))

	parameters= initialize_paramteres(n_x,n_h,n_y)
	'''
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	'''

	for i in range(0,num_iterations):

		A2,cache=forward_propagation(X,parameters)
		cost =compute_cost(A2,Y,parameters)
		grads=backward_propagation(parameters,cache,X,Y)
		parameters=update_paramters(parameters,grads,0.01)
		if print_cost and i%100==0:
			print("Cost afetr iteration %i:%f" %(i,cost))
			#print(A2.shape)
			#print(A2)
			#print(np.sum(A2,axis=0).shape)
			#print(np.sum(A2, axis=0))



	return parameters

def predict(parameters, X):
   	A2, cache = forward_propagation(X,parameters)
   	print(np.transpose(A2))
   	predictions=np.zeros(A2.shape[1])
   	predictions=np.argmax(A2,axis=0)+1
   	return predictions



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

#running the model for given number of iterations
parameters = nn_model(X, Y, 4, num_iterations=10000, print_cost=True)
'''
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))	
'''

predictions=predict(parameters,X)
print("predictions = "+str(predictions))