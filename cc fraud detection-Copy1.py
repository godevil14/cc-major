#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}',format(sys.version))
print('Numpy: {}',format(numpy.__version__))
print('Pandas: {}',format(pandas.__version__))
print('Seaborn: {}',format(seaborn.__version__))
print('scipy: {}',format(scipy.__version__))
print('Sklearn: {}',format(sklearn.__version__))
print('Matplotlib: {}',format(matplotlib.__version__))


# In[ ]:





# In[ ]:





# In[2]:


#importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[3]:


#load the dataset from csv file
data = pd.read_csv('creditcard.csv')


# In[4]:


#explore the dataset
print(data.columns
    )


# In[5]:


print(data.shape)


# In[6]:


print(data.describe
     )


# In[8]:


data = data.sample(frac = 0.1,random_state = 1)
print(data.shape
     )


# In[10]:


#plot histogram of each para
data.hist(figsize = (20,20))
plt.show()


# In[7]:


#determine number of frauds in db
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid)) 

print(outlier_fraction)





# In[9]:


corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[11]:


#get all the columns from the dataframe
columns = data.columns.tolist()

#filter the columns to remove data we dont need
columns = [c for c in columns if c not in ["Class"]]

#store the variable we'll be predicting on
target = "Class"
 
X = data[columns]
y = data[target]
print(X.shape)
print(y.shape)




# In[12]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
 
#define a random state
state = 1

#defining the outlier detection method
classifiers = {
                "Isolation Forest": IsolationForest(max_samples=len(X),
                                                    contamination = outlier_fraction,
                                                    random_state = state),
                "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors = 20,
                contamination = outlier_fraction )
}



# In[ ]:


#fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name == "Local Outlier Fac":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X) 
        
        
#reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != y).sum()
    #run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))



# In[ ]:




