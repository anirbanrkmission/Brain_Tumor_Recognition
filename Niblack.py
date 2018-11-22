
# coding: utf-8

# In[58]:


import matplotlib
import matplotlib.pyplot as plt

from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from PIL import Image
import math
import numpy as np


matplotlib.rcParams['font.size'] = 9


image = Image.open('D:/Research/Otsu.png').convert("L")
image = np.asarray(image)
#binary_global = image > threshold_otsu(image)

window_size = 25
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)

binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola

plt.figure(figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

"""
plt.subplot(2, 2, 2)
plt.title('Global Threshold')
plt.imshow(binary_global, cmap=plt.cm.gray)
plt.axis('off')
"""
plt.subplot(2, 2, 3)
plt.imshow(binary_niblack, cmap=plt.cm.Greens)
plt.title('Niblack Threshold')
plt.savefig("Nib.png")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_sauvola, cmap=plt.cm.Greens)
plt.title('Sauvola Threshold')
plt.savefig("Svl.png")
plt.axis('off')

plt.show()


# In[43]:


def Hist(img):
    row, col = img.shape 
    y = np.zeros(256)
    for i in range(0,row):
        for j in range(0,col):
            y[img[i,j]] += 1
    x = np.arange(0,256)
    plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
    plt.show()
    return y


# In[44]:


h=Hist(image)


# In[45]:


def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y


# In[46]:


def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
            cnt += h[i]
    return cnt


# In[50]:


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = np.var(np.array([0, i]))
        wb = wieght(0, i) / float(cnt)
        mb = np.mean(np.array([0, i]))
        
        vf = np.var(np.array([i, len(h)]))
        wf = wieght(i, len(h)) / float(cnt)
        mf = np.mean(np.array([i, len(h)]))
        
        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf)**2
        
        fw = open("trace.txt", "a")
        fw.write('T='+ str(i) + "\n")

        fw.write('Wb='+ str(wb) + "\n")
        fw.write('Mb='+ str(mb) + "\n")
        fw.write('Vb='+ str(vb) + "\n")
        
        fw.write('Wf='+ str(wf) + "\n")
        fw.write('Mf='+ str(mf) + "\n")
        fw.write('Vf='+ str(vf) + "\n")

        fw.write('within class variance='+ str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")
        
        if not math.isnan(V2w):
            threshold_values[i] = V2w


# In[51]:


def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    #print(list(threshold_values.values()))
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    print('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


# In[59]:


threshold_values = {}

threshold(h)
opt = get_optimal_threshold()
res = regenerate_img(image, opt)
plt.imshow(res)
plt.savefig("modified_otsu.png")

