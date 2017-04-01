import numpy
import csv
import math
import operator
import random

#accuracy funciton
def accuracy(y,pred):
	count=0.0
	for i in range(0,y.shape[0]):
		if(y[i]==pred[i]):
			count=count+1
	print (count)
	return count*100/y.shape[0]

#sigmoid function for logistic regression
def sigmoid(x,theta):
	#dot product of x and theta vectors
	no_of_rows=len(x)
	dot_product=0
	for i in range(0,no_of_rows):
		dot_product+=numpy.asscalar(x[i])*theta[i]

	val=1+math.exp(-dot_product)
	res=1/val
	return res

def lr_cost_function(feature,theta,Y,classifier,lmbda):
	no_of_rows=feature.shape[0]
	no_of_columns=feature.shape[1]
	regularized_cost=0

	for i in range(1,no_of_columns):
		regularized_cost=regularized_cost+ theta[i]**2
	
	y= numpy.zeros(no_of_rows)
	J=0
	for i in range(0,no_of_rows):
		if(Y[i]==classifier):
			y[i]=1
		else:
			y[i]=0
	for i in range(0,no_of_rows):
		hypothesis_val= sigmoid(feature[i],theta)
		J = J + -( y[i] * math.log( hypothesis_val ) + (1-y[i]) * math.log( 1- hypothesis_val )) + (lmbda/2)*regularized_cost
	
	J = J / no_of_rows 
	return J

def gradient_descent(feature,Y,theta,alpha,lmbda,classifier,iters):
	
	no_of_rows=feature.shape[0]
	no_of_columns=feature.shape[1]
	y= numpy.zeros(no_of_rows)
	hypothesis_val= numpy.zeros(no_of_rows)
	pred=numpy.zeros(no_of_rows)
	for i in range(0,no_of_rows):
		if(Y[i]==classifier):
			y[i]=1
		else:
			y[i]=0
	
	gradients=numpy.zeros(feature.shape[1])
	for k in range(0,iters):
		for i in range(0,no_of_rows):
			hypothesis_val[i]=sigmoid(feature[i],theta)
		for j in range(0,no_of_columns):
			for i in range(0,no_of_rows):
				gradients[j]=gradients[j]+(hypothesis_val[i]-y[i])*feature[i][j]

			gradients[j]=alpha*gradients[j]
			gradients[j]=gradients[j]/no_of_rows
			

			if(j!=0):
				theta[j]= theta[j] - gradients[j] + lmbda/no_of_rows*theta[j]
			else:
				theta[j]= theta[j] - gradients[j]
	if (k<10):
			print ('iteration = '+str(k)+ ' cost=' +str(lr_cost_function(feature,theta,Y,classifier,lmbda)) + '  class='+str(classifier))
	else:
			print( 'iteration = '+str(k)+ '  class='+str(classifier))

	for i in range(0,no_of_rows):
		if(sigmoid(feature[i],theta) >=0.5):
			pred[i] = 1
		else:
			pred[i] = 0
	print (pred)
	print(accuracy(y,pred))
	return theta



#create reduced feature matrix
reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)


#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)	

#appending column of 1's to feature matrix X to get new features for simplifying theta calculation
feature = numpy.ones((X.shape[0],X.shape[1]+1))
feature[:,1:] = X

no_of_classes=13
#parameters
theta = numpy.zeros((no_of_classes+1,feature.shape[1]))

for i in range (1,no_of_classes+1):
	gradient_descent(feature,Y,theta[i],0.00005,0.001,i,200)

output= numpy.dot(feature,theta[1:].transpose())

for i in range(output.shape[0]):
	for j in range(output.shape[1]):
		output[i][j]=1/(1+math.exp(-output[i][j]))


# predicted values
pred = numpy.zeros((feature.shape[0]),numpy.int)
for i in range(feature.shape[0]):
	index=0
	for j in range(1,no_of_classes):
		if(output[i][j]>output[i][index]):
			index=j
	pred[i]=index+1



print (pred)
print ('accuracy = ' + str(accuracy(pred,Y)))
