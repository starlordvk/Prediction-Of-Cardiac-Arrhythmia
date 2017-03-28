import numpy
import csv

def eucledian(x,mean,n):
	dist=0
	for i in range(0,n):
		dist=dist+(x[i]-mean[i])**2
	dist=dist**0.5
	return dist


def accuracy(y,pred):
	count=0.0
	for i in range(0,452):
		if(y[i]==pred[i]):
			count=count+1
	print (count)
	return count*100/452

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