# EEG-data-by-Machine-Learning-methods
Use Machine Learning methods to differentiate Healthy and PTSD cases/ subjects
1. Data: 
12 data subjects with 6 of Healthy (H) and 6 of PTSD (P). Each data subject is a dataset with 31 attributes (31 brain chanels) and 50000 rows (signals is measured from 31 chanels)
2. Purpose: 
Differentiate or predict H and P subjects
3. Methods:
3.1. Approach 1:
- General description: 
  + BUILD PREDICTION MODELS TO PREDICT H OR P SUBJECT
  + Use groups for data from 12 different subjects to prevent data leakage 
  + Access Cross Validation in SVM and ANN models 
- Result: Unacceptable
3.2. Approach 2: 
- General description: 
  + COMPARE THE WEIGHTS IN SVM CLASSIFICATION
  + Use Kmeans clustering to group data point in each subjects into 2 groups.
  + Use linear SVM classification to get the classification boundary based on the labels made by Kmeans (the fixed hyperlane between 2 groups) and extract the weights
  + The magnitude of the weight vectors in each SVM model of each subjects is the feature ranking and can be use to differentiate each subjects
- Result: Acceptable
