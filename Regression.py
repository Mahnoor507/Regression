#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[209]:


#training data loaded
trn_data=pd.read_csv('RegressoinTraining.csv')
x=np.array(trn_data['sales'])
y=np.array(trn_data['profit'])
trn_data.head()

#testing data loaded
test_data=pd.read_csv('RegressoinTesting.csv')
t_x=np.array(test_data['sales'])
t_y=np.array(test_data['profit'])

#data plotted on graph
plt.plot(x,y);
plt.plot(t_x,t_y);
plt.show()

#calculations
m=np.size(x)
m_x=sum(x)
m_x2=sum(x*x)
m_y=sum(y)
m_yx=sum(y*x)

#matrix
A= np.array([[m,m_x],[m_x,m_x2]])
B=np.array([[m_y],[m_yx]])

#inverse and theeta
A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

#appending 1 and predictable data
s=np.size(t_x)
new=np.zeros((s,1))
new[::]=1
t_x=t_x.reshape(s,1)
new=np.append(new,t_x,axis=1)

#predicted data
Predict_y=new.dot(theeta)
Predict_y=Predict_y.reshape(32)

#mean sqaure error
MSE=(1/(2*m))*sum((Predict_y-t_y)**2)
print("MSE : ",MSE)

#graph plotting
plt.plot(t_x,t_y)
plt.plot(t_x,Predict_y)
plt.show()


# QUADRATIC FROM NOW

# In[216]:


#training data loaded
trn_data=pd.read_csv('RegressoinTraining.csv')
x=np.array(trn_data['sales'])
y=np.array(trn_data['profit'])
trn_data.head()

#testing data loaded
test_data=pd.read_csv('RegressoinTesting.csv')
t_x=np.array(test_data['sales'])
t_y=np.array(test_data['profit'])

#data plotted on graph
plt.plot(x,y);
plt.plot(t_x,t_y);
plt.show()

#calculations
m=np.size(x)
m_x=sum(x)
m_x2=sum(pow(x,2))
m_x3=sum(pow(x,3))
m_x4=sum(pow(x,4))
m_y=sum(y)
m_yx=sum(y*x)
m_yx2=sum(y*pow(x,2))

#matrix
A= np.array([[m,m_x,m_x2],[m_x,m_x2,m_x3],[m_x2,m_x3,m_x4]])
B=np.array([[m_y],[m_yx],[m_yx2]])

#inverse and theeta
A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

#appending 1 and predictable data
s=np.size(t_x)
new=np.zeros((s,1))
new[::]=1
t_x=t_x.reshape(s,1)
t_x2=np.power(t_x,2)
new=np.append(new,t_x,axis=1)
new=np.append(new,t_x2,axis=1)

#predicted data
Predict_y=new.dot(theeta)
Predict_y=Predict_y.reshape(32)

#mean sqaure error
MSE=(1/(2*m))*sum((Predict_y-t_y)**2)
print("MSE : ",MSE)

#graph plotting
plt.plot(t_x,t_y)
plt.plot(t_x,Predict_y)
plt.show()


# CUBIC FROM NOW

# In[217]:


#training data loaded
trn_data=pd.read_csv('RegressoinTraining.csv')
x=np.array(trn_data['sales'])
y=np.array(trn_data['profit'])
trn_data.head()

#testing data loaded
test_data=pd.read_csv('RegressoinTesting.csv')
t_x=np.array(test_data['sales'])
t_y=np.array(test_data['profit'])

#data plotted on graph
plt.plot(x,y);
plt.plot(t_x,t_y);
plt.show()

#calculations
m=np.size(x)
m_x=sum(x)
m_x2=sum(pow(x,2))
m_x3=sum(pow(x,3))
m_x4=sum(pow(x,4))
m_x5=sum(pow(x,5))
m_x6=sum(pow(x,6))
m_y=sum(y)
m_yx=sum(y*x)
m_yx2=sum(y*pow(x,2))
m_yx3=sum(y*pow(x,3))

#matrix
A= np.array([[m,m_x,m_x2,m_x3],[m_x,m_x2,m_x3,m_x4],[m_x2,m_x3,m_x4,m_x5],[m_x3,m_x4,m_x5,m_x6]])
B=np.array([[m_y],[m_yx],[m_yx2],[m_yx3]])

#inverse and theeta
A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

#appending 1 and predictable data
s=np.size(t_x)
new=np.zeros((s,1))
new[::]=1
t_x=t_x.reshape(s,1)
t_x2=np.power(t_x,2)
t_x3=np.power(t_x,3)
new=np.append(new,t_x,axis=1)
new=np.append(new,t_x2,axis=1)
new=np.append(new,t_x3,axis=1)

#predicted data
Predict_y=new.dot(theeta)
Predict_y=Predict_y.reshape(32)

#mean sqaure error
MSE=(1/(2*m))*sum((Predict_y-t_y)**2)
print("MSE : ",MSE)

#graph plotting
plt.plot(t_x,t_y)
plt.plot(t_x,Predict_y)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




