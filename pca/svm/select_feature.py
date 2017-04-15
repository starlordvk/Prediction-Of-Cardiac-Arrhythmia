import csv
import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

def convert_strarr_floatarr( arr, X):
	for i in range(0,452):
		for j in range(0,278):
				X[i][j]=arr[i][j].astype(numpy.float)
	return

#Feature extraction


#create feature matrix
reader=csv.reader(open("feature.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)

#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)

	

#applying PCA to get pricipal attributes
pca = PCA(n_components=50)
X=pca.fit_transform(X)

print (pca.explained_variance_ratio_)

numpy.savetxt("reduced_features.csv",X, fmt='%s', delimiter=",")
