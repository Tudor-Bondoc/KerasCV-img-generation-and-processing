#!/usr/bin/env python
# coding: utf-8

# Bondoc Ion-Tudor, grupa 333AA
# Am lucrat in jupyter notebook si apoi am salvat fisierul ca .py

# In[1]:


# Instalare keras-cv
# Am instalat tensorflow din linia de comanda anaconda prompt
get_ipython().system('pip install --upgrade keras-cv')


# In[1]:


# Instalare suplimentara pentru import-urile de mai jos

get_ipython().system('pip install pycocotools')


# In[2]:


import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt


# In[3]:


# Obtinerea modelului conform keras.io/guides

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

images = model.text_to_image("photograph of a purple bus", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)


# In[6]:


# Salvare imagini uncompressed (BMP) conform resursei medium.com

from PIL import Image
Image.fromarray(images[0]).save("bus1.bmp")
Image.fromarray(images[1]).save("bus2.bmp")
Image.fromarray(images[2]).save("bus3.bmp")
    


# In[ ]:




