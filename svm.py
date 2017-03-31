import matplotlib.pyplot as plt
from matplotlib import style
import numpy 
style.use('ggplot')

class svm:
	def __init__(self, visualization=True):
		self.visualization=visualization
		self.colors={1:"r",-1:"b"}
		if self.visualization:
			self.fig=plt.figure()
			self.ax=self.fig.add_subplot(1,1,1)

	#fit is training
	def fit(self,data):
		self.data=data
		
		pass

	def predict(self,features):
		classification=np.sign(np.dot(np.array(features),self.w)+b)
		return classification



