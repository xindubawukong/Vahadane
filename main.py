#!/usr/bin/env python
# coding: utf-8

# # Vahadane Demo

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import spams
import cv2
import utils
from vahadane import vahadane
from sklearn.manifold import TSNE

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


SOURCE_PATH = './data/i9.png'
TARGET_PATH = './data/target1.png'
RESULT_PATH = './output/i9_to_target1_normal_f.png'


# ## Load source and target images

# In[3]:


source_image = utils.read_image(SOURCE_PATH)
target_image = utils.read_image(TARGET_PATH)
print('source image size: ', source_image.shape)
print('target image size: ', target_image.shape)
# plt.figure(figsize=(20.0, 20.0))
# plt.subplot(1, 2, 1)
# plt.title('Source', fontsize=20)
# plt.imshow(source_image)
# plt.subplot(1, 2, 2)
# plt.title('Target', fontsize=20)
# plt.imshow(target_image)
# plt.savefig(RESULT_PATH)
# plt.show()


# ## Configuration

# In[4]:


vhd = vahadane(LAMBDA1=0.1, LAMBDA2=0.01, THRESH=0.8, fast_mode=1)
vhd.show_config()


# ## Stain Separation and Color Normalization

# In[1]:


Ws, Hs = vhd.stain_separate(source_image)
Wt, Ht = vhd.stain_separate(target_image)


# In[6]:


img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
plt.figure(figsize=(20.0, 10.0))
plt.title('Result', fontsize=20)
plt.imshow(img)
plt.show()
cv2.imwrite(RESULT_PATH, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# In[7]:


tsne = TSNE(n_components=2, init='pca', random_state=19980723)
data = np.concatenate((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=0)
print(data.shape)
result = tsne.fit_transform(data.T)
print(result)


# In[8]:


data = np.concatenate((source_image[:,:,0],source_image[:,:,1],source_image[:,:,2]), axis=0)
print(data.shape)
result0 = tsne.fit_transform(data.T)
print(result0)


# In[10]:


print(result.shape)
t_min = result.min(axis=0)
t_max = result.max(axis=0)
result = (result - t_min) / (t_max - t_min)
t_min = result0.min(axis=0)
t_max = result0.max(axis=0)
result0 = (result0 - t_min) / (t_max - t_min)
print(result)
print(result0)
plt.figure(figsize=(30, 30))
plt.subplot(2,2,1)
plt.plot(result[:, 0], result[:, 1], 'r.')
plt.subplot(2,2,2)
plt.plot(result0[:,0], result0[:,1], 'r.')
plt.show()

