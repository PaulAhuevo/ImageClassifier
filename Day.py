
# coding: utf-8

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


# # Rename images in a folder.

# In[2]:

# Function to RENAME images in a folder.

def imageRename(newName):

    path = "C:/Users/pani/Desktop/Data Science/Night"  
    images = os.listdir(path)

    i = 0
    while i<len(images):
        old_file = "C:/Users/pani/Desktop/Data Science/Night" +'/' + images[i] 
        new_file = "C:/Users/pani/Desktop/Data Science/Night" +'/' + newName + "_" + str(i) + ".jpg" 
        os.rename(old_file,new_file)
        i = i+1
    
    print("Die Bilder wurden erfolgreich in " + newName + " umbenannt.")


# In[60]:

imageRename("Night")


# # Show an image

# In[5]:

# Function to SHOW an image.

def showImage (pfad):
    
    # open image
    # example URL "C:/Users/pani/Desktop/Data Science/Night/Night_8.jpg"
    img = Image.open(pfad)

    # show image
    imgplot = plt.imshow(img)


# In[36]:

showImage("C:/Users/pani/Desktop/Data Science/Night/Night_9.jpg")


# # Return avg. color

# In[8]:

# Function that RETURNS AVERAGE COLOR of an image. 

def imgToAvgColor (pfad):
    
    img = Image.open(pfad)
    img = img.resize((300,200))
    matrix = np.asarray(img)
    matrix_flat = np.reshape(matrix, matrix.size)
    
    i = 0
    color = 0
    color_avg = 1
    
    while i<matrix_flat.size:
        color = color + matrix_flat[i]
        i = i+1
        
    color_avg = color / matrix.size
    
    return color_avg

    


# In[37]:

imgToAvgColor("C:/Users/pani/Desktop/Data Science/Night/Night_0.jpg")


# # Return color vector of a folder

# In[83]:

global liste
liste = []

class NameArray:
    def __init__(self, path, trainingData):
        self.path = path
        self.trainingData = trainingData

def folder_to_matrix(pfad, nameDatei):

    # global extrem wichtig!! Sonst kein Zugriff außerhalb
    # kann das besser gelöst werden? Optimal: Name des Arrays ist ein Eingabeparameter.
    
    #global trainingData
    trainingData = []


    def img_to_color(pfad):

        # Macht Image zu avg.Color
        
        img = Image.open(pfad)
        img = img.resize((300,200))
        matrix = np.asarray(img)
        matrix_flat = np.reshape(matrix, matrix.size)

        i = 0
        color = 0
        color_avg = 1

        while i<matrix_flat.size:
            color = color + matrix_flat[i]
            i = i+1

        color_avg = color / matrix.size
        trainingData.append(color_avg)

    # Loop durch alle Bilder im Ordner
    
    j = 0
    file_count = len(os.listdir(pfad))

    while j < file_count:
        img_to_color(pfad + "/" + nameDatei + "_" + str(j) + ".jpg")
        j = j+1
    liste.append(NameArray(pfad, trainingData))
        
    



# In[84]:

folder_to_matrix("C:/Users/pani/Desktop/Data Science/Day", "Day")
print(liste[0].trainingData)
folder_to_matrix("C:/Users/pani/Desktop/Data Science/Night", "Night")
print(liste[1].trainingData)


# # Create feature vector and y     (manuel data transer needed...)

# In[85]:

#x_day = trainingData
x_day = liste[0].trainingData


# In[86]:

#x_night = trainingData
x_night = liste[1].trainingData


# In[87]:

X = x_night + x_day


# In[88]:

X = np.asarray(X)
X = X.reshape(-1,1)


# In[89]:

X


# In[90]:

y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y = np.asarray(y)


# # Train the Algorithm

# In[91]:

from sklearn.neighbors import KNeighborsClassifier 
DayNight = KNeighborsClassifier(n_neighbors=7) 
DayNight.fit(X, y)


# In[ ]:




# # Wahrsagerkugel aktivieren

# In[92]:

a = imgToAvgColor("C:/Users/pani/Desktop/Data Science/Test/buero.jpg")
prob = DayNight.predict_proba(a)
DayNight.predict_proba(90)


# In[93]:

print("Die Wahrscheinlichkeit, dass das Bild nachts aufgenommen wurde beträgt " + str((prob[0][0])*100) + " %")
print("Die Wahrscheinlichkeit, dass das Bild tagsüber aufgenommen wurde beträgt " + str((prob[0][1])*100) + " %")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[56]:

def funktion(var):
    var = [0,1]
    print(var)


# In[58]:

funktion("hallo")
hallo


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Day Folder --> Matrix with avg. color         // NICHT MEHR BENÖTIGT

# In[5]:

# Ordner --> Matrix mit avg_color     DAY

td_day = []
pfad = "C:/Users/pani/Desktop/Data Science/Day"

def img_to_color(pfad):
    
    
    img = Image.open(pfad)
    img = img.resize((300,200))
    matrix = np.asarray(img)
    matrix_flat = np.reshape(matrix, matrix.size)
    
    i = 0
    color = 0
    color_avg = 1
    
    while i<matrix_flat.size:
        color = color + matrix_flat[i]
        i = i+1
        
    color_avg = color / matrix.size
    td_day.append(color_avg)
        
    
    
td_day = []
j = 0

file_count = len(os.listdir("C:/Users/pani/Desktop/Data Science/Day"))

while j < file_count:
    img_to_color("C:/Users/pani/Desktop/Data Science/Day/Day_" + str(j) + ".jpg")
    j = j+1

td_day


# # Night Folder --> Matrix with avg. color          // NICHT MEHR BENÖTIGT

# In[46]:




# Ordner --> Matrix mit avg_color    NIGHT

td_night = []
pfad = "C:/Users/pani/Desktop/Data Science/Night"

def img_to_color(pfad):
    
    
    img = Image.open(pfad)
    img = img.resize((300,200))
    matrix = np.asarray(img)
    matrix_flat = np.reshape(matrix, matrix.size)
    
    i = 0
    color = 0
    color_avg = 1
    
    while i<matrix_flat.size:
        color = color + matrix_flat[i]
        i = i+1
        
    color_avg = color / matrix.size
    td_night.append(color_avg)
        
    
    
td_night = []
j = 0

file_count = len(os.listdir("C:/Users/pani/Desktop/Data Science/Night"))

while j < file_count:
    img_to_color("C:/Users/pani/Desktop/Data Science/Night/Night_" + str(j) + ".jpg")
    j = j+1

td_night


# In[ ]:




# In[ ]:




# ## Rename Training Data         // NICHT MEHR BENÖTIGT

# In[3]:

# rename DAY training data

array_day = os.listdir("C:/Users/pani/Desktop/Data Science/Day")

i = 0
while i<len(array_day):
    old_file = "C:/Users/pani/Desktop/Data Science/Day" +'/' + array_day[i] 
    new_file = "C:/Users/pani/Desktop/Data Science/Day" +'/' + "Day_" + str(i) + ".jpg" 
    os.rename(old_file,new_file)
    i = i+1
    



# In[154]:

# rename NIGHT training data

array_night = os.listdir("C:/Users/pani/Desktop/Data Science/Night")

i = 0
while i<len(array_night):
    old_file = "C:/Users/pani/Desktop/Data Science/Night" +'/' + array_night[i] 
    new_file = "C:/Users/pani/Desktop/Data Science/Night" +'/' + "Night_" + str(i) + ".jpg" 
    os.rename(old_file,new_file)
    i = i+1


# In[ ]:




# In[102]:

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
DayNight = KNeighborsClassifier(n_neighbors=3)
DayNight.fit(X, y) 


# In[111]:

print(DayNight.predict([[1.6]]))


# In[ ]:




# In[114]:

print(DayNight.predict_proba([[1.8]]))


# In[ ]:




# In[ ]:

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
KNeighborsClassifier(...)
print(neigh.predict([[1.1]]))
[0]
print(neigh.predict_proba([[0.9]]))
[[ 0.66666667  0.33333333]]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



