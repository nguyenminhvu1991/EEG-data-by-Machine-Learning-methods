"""
EEG DATA PROJECT
PURPOSE: DIFFERENTIATE H AND P SUBJECTS

APPROACH 2: COMPARE THE WEIGHTS IN SVM CLASSIFICATION
- Use Kmeans clustering to group data point in each subjects into 2 groups.
- Use linear SVM classification to get the classification boundary based on the labels made by Kmeans (the fixed hyperlane between 2 groups) and extract the weights
- The magnitude of the weight vector in each SVM model of each subjects is the feature ranking and can be use to differentiate each subjects

RESULT: ACCEPTABLE
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import glob


def getweights(self): 
    X= np.transpose(self)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X) #into 2 groups
    #kmeans.labels_.shape ->shape 31 for weight of 31 channels
    y=np.array(kmeans.labels_)
    clf2 = SVC(gamma='auto', kernel = 'linear')
    clf2.fit(X, y) 
    w2= abs(clf2.coef_)
    return w2

initial_path = "D:/SP19/project eeg/Python/data" #the folder contain 12 txt files, 6 files of H subjects and 6 files of P subjects
df= pd.DataFrame(index = ['H'], columns = range(1,32))
#df_ = df_.fillna(0) ->no need
for path in glob.glob(initial_path + "/*.txt"): #import  6  Healthy (H) subjects (31x50000) and 6 PTSD (P) subjects (31x50000) 
    table= pd.read_table(path)
    #X1 =np.array(table)  ->no need
    index = path[-7:-4]
    if index[0] == 'H' or index[0] == 'P':
        index = index
    else: 
        index= index[1:]
    weights = pd.DataFrame(getweights(table) , index = [index] , columns = range(1,32))
    df = pd.concat([df,weights], axis = 0)

df = df.drop('H') #the data of 31 weights for 31 chanels in 12 subjects of data

from matplotlib import pyplot as plt
for column in df:
    plt.plot(df[column] )
    plt.title('chanel  ' + str(column))
    plt.show()

'''
Check 31 graphs for 31 chanels, we can see the difference between H and P:
- chanel 4, chanel 21, chanel 19: all absolute value of weights in H chanel is higher than in P chanel, except for 1 subject.
So the accucary of this pattern is 91.6%. This is acceptable.
- chanel 12: all absolute value of weights in H chanel is higher than in P chanel. 
So the accucary of this pattern is 100%. This is acceptable.
'''

df[4]
df[12]
df[19]
df[21]
#VISUALIZATION SAMPLE
#Boxplot to compare weights in chanel 4 between H and P subjects
import matplotlib.transforms as transforms

fig, ax = plt.subplots() #can use format plt.figure(figsize=(10, 12))
H_subjects_chanel_4 = df[4][0:6]
P_subjects_chanel_4 = df[4][6:]

box = ax.boxplot([H_subjects_chanel_4, P_subjects_chanel_4], patch_artist=True,vert=1, 
                  labels=['Healthy', 'PTSD'], widths = 0.4)
colors = ['cyan', 'lightblue', ]
for patch, color in zip(box['boxes'], colors):
   patch.set_facecolor(color)
#http://www.datasciencemadesimple.com/box-plot-in-python/
y=min(H_subjects_chanel_4)
plt.axhline(y, linestyle ='--',antialiased = True)
trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(0,y, "{:.6f}".format(y), color="red", transform=trans, ha="right", va="center")
#https://stackoverflow.com/questions/42877747/add-a-label-to-y-axis-to-show-the-value-of-y-for-a-horizontal-line-in-matplotlib
plt.show()

