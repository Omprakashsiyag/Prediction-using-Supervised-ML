#!/usr/bin/env python
# coding: utf-8

# # GRIP @The Spark Foundation : Data Science & Business Analytics Intern

# # Name : Om Prakash

# # Task 01 : Prediction using Supervised ML
# 
# 
# 
# PROBLEM:
# 
# Predict the percentage of student on the basis of study hours.
# 
# What will be predict score if we a student studies for 9.5 houres/day?
# 
DATASET - 'http://bit.ly/w-data'
# # Improt Libraries

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # import the dataset

# In[15]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print('dataset import successfully!')


# In[16]:


df.describe()


# In[17]:


df.head()


# In[18]:


df.isnull()


# # Visualising Data

# In[19]:


df.plot(x='Hours',y='Scores',style='o',c='g')
plt.xlabel('Hours studies')
plt.xlabel('Score in percentage')
plt.title('Hours vs Score')
plt.show()

This graphs shows a linear relation between Hours and Score
# # Splitting the data

# In[20]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[21]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0 )
regressor = LinearRegression()
regressor.fit(x_train.reshape(-1,1), y_train)

print('training complete.')


# # Train the model

# In[22]:


line = regressor.coef_*x+regressor.intercept_


# In[23]:


plt.scatter(x, y)
plt.plot(x, line, color="red")
plt.show()


# # Making Predications

# In[29]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[30]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# # Estimating Training and test score

# In[31]:


print('Training Score:', regressor.score(x_train, y_train))


# In[32]:


print('Test Score:', regressor.score(x_test, y_test))


# # Plotting the bar graph to depict the diffrence between the actual and predicated value

# In[46]:


df.plot(kind='bar')
plt.grid(which='major', linewidth='0.5', color='orange')
plt.grid(which='minor', linewidth='0.5', color='red')
plt.show()


# # Evaluating mean absolute error

# In[51]:


from sklearn import metrics
print('Mean Absolute Eroor:', metrics.mean_absolute_error(y_test, y_pred))


# # Testing the model with our own data

# In[52]:


hours = 9.5
test = np.array([hours])
test = test.reshape(-1,1)
own_pred = regressor.predict(test)
print('No of Hours={}'.format(hours))
print('Predicted Score:{}'.format(own_pred[0]))

Therefore, the predicted score of student who studies for 9.5 hours/day is  96.16939660753593 
# # Thanks for Watching 
