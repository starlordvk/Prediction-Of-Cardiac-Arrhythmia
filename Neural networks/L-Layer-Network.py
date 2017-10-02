import numpy as np
import csv
import math
import operator
import random
import matplotlib.pyplot as plt

def relu(Z):
      return(np.maximum(Z,0),Z)

def softmax(Z):
      s=np.exp(Z)/np.sum(np.exp(Z),axis=0)	
      return(s,Z)

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

def linear_activation_forward(A_prev,W,b,activation):
      if activation=="relu":
            Z,linear_cache=linear_forward(A_prev,W,b)
            A,activation_cache=relu(Z)
      if activation=="softmax":
            Z,linear=linear_forward(A_prev,W,b)
            A,activation_cache=softmax(Z)

      assert(A.shape==(W.shape[0],A_prev.shape[1]))
      cache=(linear_cache,activation_cache)

      return A,cache

def L_model_forward(X,parameters):
	caches=[]
	A=X
	L=len(parameters)/2
	for l in range(1,L):
		A_prev=A
		A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="relu")
		caches.append(cache)
	AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="softmax")
	
def compute_cost(AL,Y):
      m=Y.shape[1]
      logprobs=(np.multiply(np.log(AL),Y))
      cost=-np.sum(logprobs)/m
      cost=float(cost)
      assert(isinstance(cost,float))
      return cost

def linear_backward(dZ,cache):
      A_prev, W, b=cache
      m=A_prev.shape[1]
      
      dW=np.dot(dZ,A_prev.T)/m
      db=np.sum(dZ,axis=1,keepdims=True)/m
      dA_prev=np.dot(W.T,dZ)
      
      assert(dA_prev.shape==A_prev.shape)
      assert(dW.shape==W.shape)
      assert(db.shape==b.shape)
      
      return dA_prev, dW, db

def linear_activation_backward_relu(dA,cache,activation):
      linear_cache, activation_cache=cache
      
      if(activation=='relu'):
            dZ=(dA)*(activation_cache["Z"]>0)
            dA_prev, dW, db=linear_backward(dZ,linear_cache)
      return dA_prev, dW, db

def linear_activation_backward_softmax(AL,Y,cache,activation):
      linear_cache, activation_cache=cache
      
      if(activation=='softmax'):
            dZ=AL-Y
            dA_prev, dW, db=linear_backward(dZ,linear_cache)
      return dA_prev, dW, db
      

def L_model_backward(AL, Y, caches):
    
      grads = {}
      L = len(caches) # the number of layers
      m = AL.shape[1]
      Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
      # Initializing the backpropagation
      ### START CODE HERE ### (1 line of code)
      #dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
      ### END CODE HERE ###
    
      # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
      ### START CODE HERE ### (approx. 2 lines)
      current_cache = caches[L-1]
      grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_softmax(AL,Y, current_cache, activation = "softmax")
      ### END CODE HERE ###
    
      for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_relu(grads["dA"+str(L)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

      return grads
      
def update_parameters(parameters, grads, learning_rate):
    
      L = len(parameters)/ 2 # number of layers in the neural network

      # Update rule for each parameter. Use a for loop.
      ### START CODE HERE ### (≈ 3 lines of code)
      for l in range(L):
            parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
            parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
      ### END CODE HERE ###
        
      return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
      costs = []                         # keep track of cost
    
      # Parameters initialization.
      ### START CODE HERE ###
      parameters = initialize_parameters_deep(layers_dims)
      ### END CODE HERE ###
    
      # Loop (gradient descent)
      for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(X, parameters)
            ### END CODE HERE ###
        
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(AL, Y)
            ### END CODE HERE ###
    
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
 
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                  print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                  costs.append(cost)
            
      # plot the cost
      plt.plot(np.squeeze(costs))
      plt.ylabel('cost')
      plt.xlabel('iterations (per tens)')
      plt.title("Learning rate =" + str(learning_rate))
      plt.show()
    
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

