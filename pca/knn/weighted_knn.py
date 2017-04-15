import numpy
import csv
import math
import operator
import random
from sklearn.neighbors import KNeighborsClassifier


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

clf = KNeighborsClassifier(n_neighbors=13,weights='distance')

print
print("..................................Traing set................................")
print
clf.fit(X_train, Y_train)
print(clf.predict(X_train))
score = clf.score(X_train, Y_train)
print "Training Set size = ", c
print "Training Set accuracy = ", score*100
print "Training Set error = ", (1-score)*100


print
print("..................................Test set...................................")
print
print(clf.predict(X_test))
score = clf.score(X_test, Y_test)
print "Test Set size = ",X.shape[0]-c
print "Test Set accuracy = ", score*100
print "Test Set error = ",(1-score)*100