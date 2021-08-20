#!/usr/bin/env python
# coding: utf-8

# # Compressed Signal Reconstruction Using StOMP for EEG Signal
# ### Cantika Puspa Karuniaputri - Telecommunication Engineering - School of Electrical Engineering
# 

# * Data aquisition using Neurosky MindWave Mobile 2 and Aruino Uno + HC-05 module
# * Data reconstruction using Raspberry Pi 
# * Successfully simulated on Matlab with compression ratio 75% with 64 data, MAE approx. 10%

# In[1]:


import numpy as np
from numpy import linalg as LA
import scipy.io as sio
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import sys
np.set_printoptions(threshold=sys.maxsize)


# ## DATASET

# In[2]:


original = pd.read_excel ('eeg_data.xlsx') #original data
compressed = np.genfromtxt('compressed.txt', dtype = complex, delimiter = ',') #compressed signal
toeplitz = np.genfromtxt('BT4864.txt', dtype = None, delimiter = ',') #transfomation matrix
print(toeplitz)


# ## Reconstruction Process

# In[24]:


A = toeplitz
data = 64
N = 64
#pre requitment
y = compressed
r = y
x_pre = np.zeros(N)
o = []
i = 0 #iteration
t = 10 #threshold
sd = LA.norm(r)

    
# Control stop interation with norm thresh or sparsity
#while i<data:

    # Compute the score of each atoms
c = np.dot (A.T, r)
absValues = np.abs (c)
absValues
        
    #noise level (standard deviation)
sd = LA.norm(r) / math.sqrt(data)

        
    #find the desired indices greater
tsd = t*sd
tsd
a = np.where(absValues<=tsd)
ind = c[a]      
o = np.union1d(o, ind)
size = o.shape
Ao=A[:,size:]
Ao
#x_pre[size] = np.linalg.inv(np.dot(Ao.T,Ao)).dot(Ao.T).dot(y)
    #r = y - A.dot(x_pre)
            
   # i += 1
    #print (x_pre[o])


# In[ ]:




