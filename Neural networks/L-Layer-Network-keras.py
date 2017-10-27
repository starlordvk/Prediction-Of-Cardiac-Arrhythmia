# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 07:30:17 2017

@author: VARUN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

#importing the dataset
X=pd.read_csv("reduced_features.csv")
X=X.iloc[:,:].values
Y=pd.read_csv("target_output.csv")
Y=Y.iloc[:,:].values


#encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
Y=onehotencoder.fit_transform(Y).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.05, random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

#initialising the ann
classifier= Sequential()

#adding input layer and 1st hidden layer
classifier.add(Dense(output_dim=10,activation='relu',init='uniform',input_dim=175))

#adding second hidden layer
classifier.add(Dense(output_dim=10,activation='relu',init='uniform'))

#adding third hidden layer
classifier.add(Dense(output_dim=10,activation='relu',init='uniform'))

#adding output layer
classifier.add(Dense(output_dim=13,activation='softmax',init='uniform'))


#compiling ANN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
classifier.fit(X_train,Y_train,batch_size=25,epochs=200)



#predicting test sets results
#Y_pred=classifier.predict(X_test)
#Y_test_op=np.argmax(Y_pred,axis=1)+1





