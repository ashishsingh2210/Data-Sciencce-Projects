#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[28]:


(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()


# In[29]:


len(x_train)


# In[30]:


len(y_train)


# In[31]:


len(x_test)


# In[32]:


len(y_test)


# In[33]:


x_train[0].shape


# In[34]:


x_train[0]


# In[35]:


plt.matshow(x_train[0])


# In[36]:


plt.matshow(x_train[2])


# In[37]:


y_train[2]


# In[38]:


y_train[0:5]


# In[39]:


x_train.shape


# In[40]:


# flattening the data -> converting 2d array into 1d array

x_train_fattened = x_train.reshape(len(x_train),28*28)
x_test_fattened = x_test.reshape(len(x_test),28*28)
print(x_test_fattened.shape)
x_train_fattened.shape


# In[41]:


x_train_fattened[0]


# In[42]:


# create simple neural network


# In[43]:


model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train_fattened,y_train,epochs=5)


# In[44]:


# scaling -> cranging data between 0 to 1 

x0_train = x_train / 255
x0_test = x_test / 255

x0_train_fattened = x0_train.reshape(len(x0_train),28*28)
x0_test_fattened = x0_test.reshape(len(x0_test),28*28)
print('x0_test_fattened_shape - ',x0_test_fattened.shape)
print('x0_train_fattened_shape - ',x0_train_fattened.shape)

x0_train_fattened[0]


# In[45]:


# create simple neural network

model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x0_train_fattened,y_train,epochs=5)


# ###### above you can see hoe scaling is more important for better accuracy

# In[46]:


# evalutationg accuracy in test dataset without scaling
model.evaluate(x_test_fattened,y_test)


# In[49]:


# evalutationg accuracy in test dataset with scaling

model.evaluate(x0_test_fattened,y_test)


# In[51]:


plt.matshow(x0_test[0])


# In[52]:


y0_predicted = model.predict(x0_test_fattened)
y0_predicted[0]


# In[53]:


np.argmax(y0_predicted[0])


# In[54]:


np.argmax(y0_predicted[1])


# In[72]:


y0_predicted_labels = [np.argmax(i) for i in y0_predicted]
y0_predicted_labels[:5]


# In[71]:


y_test[:5]


# In[79]:


sess = tf.Session()


# In[81]:


# confusion matrix

cm = tf.math.confusion_matrix(y_test,y0_predicted_labels)
cm = cm.eval(session=sess)
cm


# In[83]:


import seaborn as sns

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# In[86]:


# create simple neural network with hidden layer

model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x0_train_fattened,y_train,epochs=5)


# In[87]:



model.evaluate(x0_test_fattened,y_test)


# In[90]:


# create simple neural network with hidden layer

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x0_train,y_train,epochs=5)


# In[ ]:




