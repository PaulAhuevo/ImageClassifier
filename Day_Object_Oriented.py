
# coding: utf-8

# # Import Libraries

# In[1]:

import pandas as pd
import numpy as np
import pylab as pl
import PIL
from PIL import Image
import os
import base64
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import StringIO


from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

get_ipython().magic('matplotlib inline')


# # Image Processing

# In[72]:

# example URL "C:/Users/pani/Desktop/Data Science/Night/Night_8.jpg"

def image_rename (newName):
   
   path = "C:/Users/pani/Desktop/Data Science/forest"  
   images = os.listdir(path)

   i = 0
   while i<len(images):
       old_file = "C:/Users/pani/Desktop/Data Science/forest" +'/' + images[i] 
       new_file = "C:/Users/pani/Desktop/Data Science/forest" +'/' + newName + "_" + str(i) + ".jpg" 
       os.rename(old_file,new_file)
       i = i+1
   
   
def show_image (pfad):
   
   img = Image.open(pfad)
   imgplot = plt.imshow(img)


def img_to_color (pfad):
   
   img = Image.open(pfad)
   img = img.resize((300,200))
   matrix = np.asarray(img)
   matrix_flat = np.reshape(matrix, matrix.size)
   
   i = 0
   color = 0
   
   global color_avg    # Wieso bruache ich hier global wenn ich return habe?
   color_avg = 1
   
   while i<matrix_flat.size:
       color = color + matrix_flat[i]
       i = i+1
   
   color_avg = color / matrix.size
   
   return color_avg


# In[74]:

image_rename("forest")


# In[4]:

def img_to_rgb (pfad):

    # pfad = "C:/Users/pani/Desktop/Data Science/Night/Night_8.jpg"

    img = Image.open(pfad)
    img = img.resize((300,200))

    pixel_red = []
    pixel_green = []
    pixel_blue = []

    pixels = list(img.getdata())
    num_pixels = len(pixels)
    i = 0

    while i<60000:
        pixel_red.append(pixels[i][0])
        i = i+1

    i = 0

    while i<60000:
        pixel_green.append(pixels[i][1])
        i = i+1

    i = 0

    while i<60000:
        pixel_blue.append(pixels[i][2])
        i = i+1

    red = sum(pixel_red) / len(pixel_red)
    green = sum(pixel_green) / len(pixel_green)
    blue = sum(pixel_blue) / len(pixel_blue)
    
    return([red, green, blue])


# In[5]:

img_to_rgb("C:/Users/pani/Desktop/Data Science/Night/Night_1.jpg")


# # Return color vector of a folder

# In[78]:

# "C:/Users/pani/Desktop/Data Science/Day", "Day"

def folder_to_matrix(pfad, nameDatei):

    trainingData = []
    file_count = len(os.listdir(pfad))
    j = 0

    while j < file_count:
        img_to_color(pfad + "/" + nameDatei + "_" + str(j) + ".jpg")
        trainingData.append(color_avg)
        j = j+1
        
    return trainingData

def folder_to_rgb(pfad, nameDatei):

    red_value = []
    green_value = []
    blue_value = []
    
    file_count = len(os.listdir(pfad))
    j = 0
    RGB = []

    while j < file_count:
        
        imagergb = img_to_rgb(pfad + "/" + nameDatei + "_" + str(j) + ".jpg")
        
        
        
        RGB.append(imagergb)
        
        j = j+1
        
    return RGB


# # Preparing Dataset

# In[147]:

# Preparing RGB Dataset

night = folder_to_rgb("C:/Users/pani/Desktop/Data Science/Night", "Night")
beach = folder_to_rgb("C:/Users/pani/Desktop/Data Science/beach", "beach")
forest = folder_to_rgb("C:/Users/pani/Desktop/Data Science/forest", "forest")

y = [[0] * len(night) + [1] * len(beach) + [2] * len(forest)]
y = y[0]

x = [night + beach + forest]
x = x[0]

print(len(x))
print(len(y))


# In[86]:

# Preparing the brightness Dataset

x_DAY = folder_to_matrix("C:/Users/pani/Desktop/Data Science/Day", "Day")
x_NIGHT = folder_to_matrix("C:/Users/pani/Desktop/Data Science/Night", "Night")
x_DAWN = folder_to_matrix("C:/Users/pani/Desktop/Data Science/Dawn", "Dawn")

y_DAY = [2] * len(x_DAY)
y_DAWN = [1] * len(x_DAWN)
y_NIGHT = [0] * len(x_NIGHT)

x = x_DAY + x_DAWN + x_NIGHT
y = y_DAY + y_DAWN + y_NIGHT

x = np.asarray(x)
x = x.reshape(-1,1)
y = np.asarray(y)


# # Train the Algorithm

# In[215]:

from sklearn.neighbors import KNeighborsClassifier 
Hellseher = KNeighborsClassifier(n_neighbors=15) 
Hellseher.fit(x, y)


# # Classify Images

# In[210]:

test = img_to_rgb("C:/Users/pani/Desktop/Data Science/test/test3.jpg")
test = np.asarray(test)
test = test.reshape(1, -1)

print(test)


# In[211]:

z = Hellseher.predict_proba(test)


# In[214]:

print("Wahrscheinlichkeit Nacht: " + str((z[0][0])*100)[:5] + " %")
print("Wahrscheinlichkeit Strand: " + str((z[0][1])*100)[:5] + " %")
print("Wahrscheinlichkeit Wald: " + str((z[0][2])*100)[:5] + " %")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



