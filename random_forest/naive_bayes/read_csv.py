import csv
import numpy
def mean_column( X, col_num ):
	mean=0.0
	c=0
	for i in range(0,452):
		if(X[i][col_num] !="?"):	
			mean=mean+X[i][col_num].astype(numpy.float)
			c=c+1
	mean=mean/c
	return mean

def standard_deviation_column( X, col_num ,mean):
	sd=0.0
	c=0
	for i in range(0,452):
		if(X[i][col_num] !="?"):	
			sd=(X[i][col_num].astype(numpy.float)-mean)**2
			c=c+1
	sd=sd/(c-1)
	sd=sd**0.5
	return sd
	
	
def convert_strarr_floatarr( arr, X):
	for i in range(0,452):
		for j in range(0,278):
			if(arr[i][j]=="?"):	
				X[i][j]=0.0
			else:
				X[i][j]=arr[i][j].astype(numpy.float)
	return
					
			
reader=csv.reader(open("arrhythmia.csv","r"),delimiter=",")
arr=list(reader)
arr=numpy.array(arr)
data=numpy.zeros((452,2))
c=0
for i in range(0,452):
	for j in range(0,279):
		if(arr[i][j] =="?"):	
			data[c][0]=i
			data[c][1]=j
			c=c+1

#majority of the values are missing so delete coulmn 13
#find the columns with missing values			
for i in range(0,c):
	if(data[i][1]!=13):
		print(data[i][0],data[i][1])

#remove coulmn 13
arr = numpy.delete(arr,13,1)

#create feature matrix
X=numpy.zeros((452,278),dtype=numpy.float)
convert_strarr_floatarr(arr,X)

#create result vector
y=numpy.zeros((452),dtype=numpy.int)
for i in range(0,452):
	y[i]=arr[i][278].astype(numpy.int)
print y

#find the columns with missing values			
for i in range(0,c):
	if(data[i][1]!=13):
		print(data[i][0],data[i][1])
			
#calculate mean for column 13(initially 14),11,10,12
mean=mean_column(X,13)
print "mean="+str(mean)
sd=standard_deviation_column(X,13,mean)

for i in range(0,452):
	if(arr[i][13]=="?"):
		val = numpy.random.normal(mean,sd,1)
		print val
		X[i][13]=(val).astype(numpy.int)
		print X[i][13]
		
mean=mean_column(X,10)
print "mean="+str(mean)
sd=standard_deviation_column(X,10,mean)

for i in range(0,452):
	if(arr[i][10]=="?"):
		val = numpy.random.normal(mean,sd,1)
		print val
		X[i][10]=(val).astype(numpy.int)
		print X[i][10]

mean=mean_column(X,11)
print "mean="+str(mean)
sd=standard_deviation_column(X,11,mean)

for i in range(0,452):
	if(arr[i][11]=="?"):
		val = numpy.random.normal(mean,sd,1)
		print val
		X[i][11]=(val).astype(numpy.int)
		print X[i][11]

mean=mean_column(X,12)
print "mean="+str(mean)
sd=standard_deviation_column(X,12,mean)

for i in range(0,452):
	if(arr[i][12]=="?"):
		val = numpy.random.normal(mean,sd,1)
		print val
		X[i][12]=(val).astype(numpy.int)
		print X[i][12]
#reduce number of classes
for i in range(0,452):
	if (y[i]>=14):
		y[i]=y[i]-3

numpy.savetxt("feature.csv", X, fmt='%s', delimiter=",")
numpy.savetxt("target_output.csv", y, fmt='%s', delimiter=",")
#result=numpy.array(x).astype("str")
