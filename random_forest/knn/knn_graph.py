import numpy
import csv
import math
import operator
import random
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def loadDataset(split,X,Y, X_train=[] , Y_train=[],  X_test=[],  Y_test=[]):
    c=0
    for i in range(0,X.shape[0]):
        if random.random() < split:
            X_train.append(X[i])
            Y_train.append(Y[i])
            c=c+1
        else:
            X_test.append(X[i])
            Y_test.append(Y[i])
    return c



reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)

#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)
Y=Y.ravel()

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]


c=loadDataset(0.8,X,Y, X_train , Y_train,  X_test,  Y_test)

graph_x=[]
graph_y_train=[]
graph_y_test=[]
num=100
for i in range(1,num+1):
    clf = KNeighborsClassifier(n_neighbors=i,weights='distance')
    clf.fit(X_train, Y_train)
    
    graph_x.append(i)
    score = clf.score(X_train, Y_train)
    graph_y_train.append(score)
    score = clf.score(X_test, Y_test)
    graph_y_test.append(score)



plt.ylabel('accuracy')
plt.xlabel('weighted k nearest neighbors')
plt.plot(graph_x,graph_y_train,'r')
plt.plot(graph_x,graph_y_test,'b')
plt.show()
