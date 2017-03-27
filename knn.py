import numpy
import csv




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

class_count=numpy.zeros(14,dtype=numpy.int)

c=0
for i in range (0,452):
	z=Y[i].astype(int)
	for j in range (0,no_of_columns):
		print z
		mean_matrix[z][j]=4#mean_matrix[z][j]+X[i][j]
		print mean_matrix[z][j]
		class_count[z]=class_count[z]+1;
		print j
		#print z
for i in range (1,14):
	for j in range (0,no_of_columns):
		mean_matrix[i][j]=mean_matrix[i][j]/class_count[i]
	



