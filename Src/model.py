#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# In[9]:


data = pd.read_csv(r'C:\Users\x\Documents\program\Python\Project\Asset\cardio_train.csv', sep=';')
data


# In[10]:


X = data.drop(['id', 'cardio'], axis=1)
X['age'] = (X['age'] / 365).round().astype(int)
X['gender'] = X['gender'].map({1: 0, 2: 1})
y = data['cardio']


# In[13]:


X


# In[14]:


dataflow = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('estimator', LogisticRegression())
])  


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[16]:


model = dataflow.fit(X_train, y_train)


# In[21]:


from sklearn.metrics import accuracy_score
score = model.score(X_test, y_test)
predict = model.predict(X_test)
accuracy = accuracy_score(y_test, predict)
accuracy, score

def prediction_model(data):
    #data -> age gender height weight(float) ap_hi ap_lo cholesterol gluc smoke alco active
    import numpy as np
    sample = np.array([data])
    result = model.predict(sample)[0]
    return "high risk" if result == 1 else "low risk"

