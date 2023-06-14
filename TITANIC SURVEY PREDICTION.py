#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# In[2]:


#Data Collection & Processing

# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv('C:/Users/206277/Documents/train.csv')


# In[3]:


# printing the first 5 rows of dataFrame
titanic_data.head()


# In[4]:


# printing the last 5 rows of dataFrame
titanic_data.tail()


# In[5]:


# number of rows and columns
titanic_data.shape


# In[6]:


# getting some informations about the data
titanic_data.info()


# In[7]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# In[8]:


# drop the cabin table because of more missing values
# drop the "cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)


# In[9]:


# replace the missing values in "Age" column with mean value 
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[10]:


# finding  the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())


# In[11]:


# replacing the missing values in "Embarked" column with mode value
titanic_data["Embarked"].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[12]:


titanic_data.isnull().sum()


# In[13]:


#Data Analysis

# getting some statistical measures about the data
titanic_data.describe()


# In[14]:


#making countplot for "sex" column
titanic_data['Sex'].value_counts()


# In[15]:


titanic_data['Embarked'].unique()


# In[16]:


# Encoding the Categorical column
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[17]:


titanic_data.head()


# In[18]:


#Separating features and target
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[31]:


print(X)


# In[20]:


print(Y)


# In[22]:


#splitting the data into training data & Test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)



# In[23]:


#Model Training:
model = LogisticRegression()


# In[24]:


# training the logistic Regression model with training data
model.fit(X_train,Y_train)


# In[25]:


# accuracy on training data
X_train_prediction = model.predict(X_train)


# In[26]:


print(X_train_prediction)


# In[27]:


#model evaluation:
#Accuracy Score
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[28]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[29]:


print(X_test_prediction)


# In[30]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




