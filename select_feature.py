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

	

#applying random forests to get pricipal attributes
model = ExtraTreesClassifier()
model.fit(X, Y.ravel())
#print(model.feature_importances_)

numpy.savetxt("randforrests.csv", model.feature_importances_, fmt='%s', delimiter=",")

#selecting features 
c=0;
important_features=numpy.zeros((278),dtype=numpy.float)
important_features_index=numpy.zeros((278),dtype=numpy.int)

for i in range (0,278):
	if((model.feature_importances_[i]*1000)>=1.0):
		important_features[c]=model.feature_importances_[i]
		important_features_index[c]=i
		c=c+1

print(important_features)	
print(important_features_index)
print("The no of features =",c)	

#features are reduced  from 278

numpy.savetxt("import_features_index_after_random_forrests.csv",important_features_index, fmt='%s', delimiter=",")

#new matrix compirising of reduced features
newX=numpy.zeros((452,c),dtype=numpy.float)
for i in range (0,452):
	for j in range (0,c):
		newX[i][j]=X[i][important_features_index[j]]


print(newX)

numpy.savetxt("reduced_features.csv",newX, fmt='%s', delimiter=",")
