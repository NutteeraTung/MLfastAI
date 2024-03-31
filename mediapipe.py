#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys


# In[3]:


get_ipython().system('{sys.executable} -m pip install gradio mediapipe opencv-python matplotlib')


# In[4]:


import cv2
import mediapipe as mp
import numpy as np


# In[6]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow('Selfie Seg', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()


# In[7]:


mp_selfie = mp.solutions.selfie_segmentation


# In[12]:


cap = cv2.VideoCapture(0)
# Create with statement for model 
with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Apply segmentation
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.process(frame)
        frame.flags.writeable = True

        cv2.imshow('Selfie Seg', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[13]:


res.segmentation_mask


# In[14]:


from matplotlib import pyplot as plt
from matplotlib import gridspec


# In[15]:


# Layout
plt.figure(figsize=(15,15))
grid = gridspec.GridSpec(1,2)

# Setup axes
ax0 = plt.subplot(grid[0])
ax1 = plt.subplot(grid[1])

ax0.imshow(frame)
ax1.imshow(res.segmentation_mask)
plt.show()


# In[16]:


background = np.zeros(frame.shape, dtype=np.uint8)
mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 


# In[17]:


np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 


# In[18]:


segmented_image = np.where(mask, frame, background)


# In[19]:


plt.imshow(segmented_image)


# In[20]:


# Layout
plt.figure(figsize=(15,15))
grid = gridspec.GridSpec(1,2)

# Setup axes
ax0 = plt.subplot(grid[0])
ax1 = plt.subplot(grid[1])

ax0.imshow(res.segmentation_mask)
ax1.imshow(segmented_image)
plt.show()


# In[21]:


segmented_image = np.where(mask, frame, cv2.blur(frame, (40,40)))


# In[22]:


# Layout
plt.figure(figsize=(15,15))
grid = gridspec.GridSpec(1,2)

# Setup axes
ax0 = plt.subplot(grid[0])
ax1 = plt.subplot(grid[1])

ax0.imshow(res.segmentation_mask)
ax1.imshow(segmented_image)
plt.show()


# In[23]:


import gradio as gr


# In[24]:


def segment(image): 
    with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 
        return np.where(mask, image, cv2.blur(image, (40,40)))


# In[25]:


webcam = gr.inputs.Image(shape=(640, 480), source="webcam")


# In[26]:


webapp = gr.interface.Interface(fn=segment, inputs=webcam, outputs="image")


# In[27]:


webapp.launch()


# In[ ]:




