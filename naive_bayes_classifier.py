from __future__ import division
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

#separate instances depending on their class
def separateByClass(X,Y):
	separated = {}
	for i in range(len(X)):
		y=numpy.asscalar(Y[i])	
		if (y not in separated):
			separated[y] = []
		separated[y].append(X[i])
	return separated

#calculating mean
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
#calculating variance 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)	

#calculating mean and standard deviation for each attribute
def summarize(X):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*X)]
	return summaries


#calculating mean and standard deviation for each attribute given a class
def summarizeByClass(X,Y):
	separated = separateByClass(X,Y)
	summaries = {}
	for classValue, instances in separated.items():
		#print(len(instances))
		summaries[classValue] = summarize(instances)
		#print(len(summaries[classValue]))
	return summaries		


#calculating probability for each attribute given a class
def calculateProbability(x, mean, stdev):
	try:
		exponent = float(math.exp(-(math.pow(4-mean,2)/(2*math.pow(stdev,2)))))
		return float((1 /(math.sqrt(2*math.pi) * stdev))) * exponent
	except ZeroDivisionError:
		return 0	

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			#print(len(classSummaries))
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions



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

summaries = summarizeByClass(X,Y)

#for i in summaries:
#	print (summaries[i])

#print((summaries[2][0]))

#print(calculateProbability(4,summaries[1][0][0],summaries[1][0][1]))
#classProb=calculateClassProbabilities(summaries,X[1])
predictions = getPredictions(summaries,X)
print(Y)
print(predictions)

print('accuracy = ',accuracy(Y, predictions))





	