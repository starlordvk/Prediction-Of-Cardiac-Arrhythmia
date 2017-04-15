import numpy
import csv
import math
import operator
import random
'''
def eucledian(x,mean,n):
	dist=0
	for i in range(0,n):
		dist=dist+(x[i]-mean[i])**2
	dist=dist**0.5
	return dist

'''
def accuracy(y,pred):
	count=0.0
	for i in range(0,452):
		if(y[i]==pred[i]):
			count=count+1
	print (count)
	return count*100/452




def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((x, dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors,Y):
	classVotes = {}
	for x in range(len(neighbors)):
		response =numpy.asscalar(Y[neighbors[x]]);
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions,Y):
		correct = 0
		for x in range(len(testSet)):
			if Y[x] == predictions[x]:
				correct += 1
				return (correct/float(len(testSet))) * 100.0


			
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

no_of_columns=X.shape[1]


predictions=[]
k = 6

#get k nearest neighbors
for i in range(0,452):
	neighbors = getNeighbors(X,X[i],k)
	result = getResponse(neighbors,Y)
	predictions.append(result)
print (predictions)
print ('accuracy=' +str(accuracy(Y,predictions)))
#print('> predicted=' + repr(result) + ', actual=' + repr(X[120][-1]))
#accuracy = getAccuracy(X[120], predictions,Y)
#print('Accuracy: ' + repr(accuracy) + '%')

'''
#matrix to store the mean values for each class
mean_matrix=numpy.zeros((14,no_of_columns),dtype=numpy.float)
print (mean_matrix.shape)

class_count=numpy.zeros(14,dtype=numpy.int)


for i in range (0,452):
	z=numpy.asscalar(Y[i])
	for j in range (0,no_of_columns):
		mean_matrix[z][j]=mean_matrix[z][j]+X[i][j]
	class_count[z]=class_count[z]+1

for i in range (1,14):
	for j in range (0,no_of_columns):
		mean_matrix[i][j]=mean_matrix[i][j]/class_count[i]

#predicted value arrays
pred=numpy.zeros((452,),dtype=numpy.int)

#KNN
for i in range(0,452):
	dist=eucledian(X[i],mean_matrix[1],no_of_columns)
	pred[i]=1
	for j in range(2,14):
		temp=eucledian(X[i],mean_matrix[j],no_of_columns)
		#print temp
		#print dist,temp
		if(dist>temp):
			pred[i]=j
			dist=temp

print (pred)
print (Y.transpose())
print (accuracy(Y,pred))

correct_class_count=numpy.zeros(14,dtype=numpy.int)
for i in range(0,452):
	z=numpy.asscalar(Y[i])
	if(pred[i]==Y[i]):
			correct_class_count[z]=correct_class_count[z]+1

for i in range(1,14):
	print ("class= ",str(i),str(correct_class_count[i]))
	'''