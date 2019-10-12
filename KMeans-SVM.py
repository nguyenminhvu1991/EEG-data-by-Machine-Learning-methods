#USE COMPARISON OF WEIGHTS APPROACH 
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans

path = r'D:\SP19\project eeg\Python\P\P11.txt'
ch1 = open(path)
table = pd.read_table(path)

#KMEANS CLUSTERING 1

X = np.array(table)
type(X)
kmeans = KMeans(n_clusters=6, random_state=0).fit(X) #into 6 groups
kmeans.labels_
X_dataframe =pd.DataFrame(kmeans.labels_)
X_dataframe.index.names=['CHANNEL']
X_dataframe.rename( columns={0:'GROUP'}, inplace=True)
X_dataframe
#GET THIS ONE TO COMPARE
#????WHICH CHANEL INTO WHICH GROUP

"""#SVM TRIAL 
#=> not good for comparing
y = np.array(kmeans.labels_)

clf = SVC(gamma='auto', kernel = 'linear')
clf.fit(X, y) 
w= clf.coef_
#w.shape
#(15, 50000)
#15 = 2C6 => NOT GOOD TO COMPARE"""


# KMEANS CLUSTERING 2 
#FOR SVM LABEL, SUPPOSE COMBINATION OF SIGNAL 1 IS AN OBJECT, SET OF SIGNAL 1 IN 31 CHANELS MAY BE SIMULTANEOUSLY MEASURED
X2= np.transpose(table)
kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(X2) #into 2 groups
kmeans_2.labels_.shape
kmeans_2.labels_
y2=np.array(kmeans_2.labels_)
clf2 = SVC(gamma='auto', kernel = 'linear')
clf2.fit(X2, y2) 
w2= clf2.coef_
#shape of 31 for weight of 31 channels
#GET THIS ONE TO COMPARE
# channel number 21 (FC1)