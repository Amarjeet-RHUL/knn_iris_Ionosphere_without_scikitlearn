#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT OBSERVATIONS & NOTES

# In[263]:


"""
IRIS & IONOSPHERE DATASET:

-    Implemented the K Nearest Neighbours algorithm for a general K but used 1 & 3 instead of K in the programs.

-    Normalization can be used to rescale the data in the range of 0 to 1. I have commented the codes in 7th CELL.
It ll be used for test datset n we can find the values similarly.

-    We can evaluate the program by using K-Fold Validation.

-    Plots can be used to generate the graphical reprentation.

-    Genralized k value we can use the below commented function by adding the 2 lines which will ask for an iput integer value of k in KNN:
Added as a comment line in my program
# k=int(input("enter the k value of KNN"))
# y_pred_data = KNN_Neighbors(new_train_data, new_test_data, k)

-   

"""


# # KNN ALgorithm for IRIS_DATASET(General K value(Commented) & 1-NN
# 

# In[264]:



from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
iris.keys()


# In[265]:


iris["data"], iris['target']


# In[268]:


#Split the Dataset using DOB as random_state
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=705)


# In[269]:


print(len(X_train), len(X_test), len(y_train), len(y_test))

y_test


# In[270]:


new_train_data = []
for i in range(len(X_train)):
    print(X_train[i])
    print(y_train[i])
    new_train_data.append(np.append(X_train[i], y_train[i]).tolist())
    print(new_train_data)


# In[271]:


new_test_data = []
for i in range(len(X_test)):
    new_test_data.append(np.append(X_test[i], y_test[i]).tolist())
    print(X_test[i])
    print(y_test[i])
    print(new_test_data)
    


# In[272]:


# Rescaling & finding the max n minimum value:
# def iris_mnmx(dataset):
#     mxmn = list()
#     for i in range(len(dataset[0])):
#         col_values = [row[i] for row in dataset]
#         value_min = min(col_values)
#         value_max = max(col_values)
#         mxmn.append([value_min, value_max])
#     return mxmn

# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, mxmn):
#     for row in dataset:
#         for i in range(len(row)):
#             row[i] = (row[i] - mxmn[i][0]) / (mxmn[i][1] - mxmn[i][0])
      


# normalize_dataset(new_train_data,iris_mnmx(new_train_data))
# print(new_train_data)


# In[273]:


print(len(new_test_data),len(new_train_data))


# In[274]:


#Normalization is not used but can be implemented if required
#Used merge sort for sorting the distances & correspondig dataset
def mergeSort(nlist):
    #print("Splitting ",nlist)
    if len(nlist)>1:
        mid = len(nlist)//2
        lefthalf = nlist[:mid]
        righthalf = nlist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=j=k=0       
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i][1] < righthalf[j][1]:
                nlist[k]=lefthalf[i]
                i=i+1
            else:
                nlist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            nlist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            nlist[k]=righthalf[j]
            j=j+1
            k=k+1
            
        
#Finding the Euclidean distanct datapoints/vectors
def distance_Eucl(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance =distance + ((point1[i] - point2[i])**2)
    return sqrt(distance)
 
# Find neighbours
def neighbours_search(train_data, rows_test, knn_neighbours):
    distances = list()
    for rows_train in train_data:
        dist = distance_Eucl(rows_test, rows_train)
        distances.append((rows_train, dist))
    mergeSort(distances)    
    neighbors = list()
    for i in range(knn_neighbours):
        neighbors.append(distances[i][0])
    return neighbors
 
# Doing prediction with the neighbours
def classification_pred(train_data, rows_test, knn_neighbours):
    neighbors = neighbours_search(train_data, rows_test, knn_neighbours)
    op_val = [line[-1] for line in neighbors]
    prediction_val = max(set(op_val), key=op_val.count)
    return prediction_val
 
# KNN Algorithm
def KNN_Neighbors(train_data, test, knn_neighbours):
    preds = list()
    for row in test:
        output = classification_pred(train_data, row, knn_neighbours)
        preds.append(output)
#         print(predictions)
    return(preds)


            


# In[276]:


#for genralized k value we can use the below commented function:


# k=int(input("enter the k value of KNN"))
# y_pred_data = KNN_Neighbors(new_train_data, new_test_data, k)

y_pred_data = KNN_Neighbors(new_train_data, new_test_data, 1)
y_pred = []

y_pred = y_pred_data
np.array(y_pred), y_test

# print(len(y_pred) , len(y_test))
# print(y_pred , y_test)
count=0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        count=count+1
# print(count)
Acc=count/len(y_pred)
print("Accuracy is :", Acc)
Err_Rate= 1-Acc
print("Error Rate is :",Err_Rate)


# # IRIS_Dataset & 3-NN Algorithm

# In[277]:


"""
Implemented the K Nearest Neighbours algorithm for a general K but used 1 & 3 instead of K in the programs
 
"""


from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
iris.keys()


# In[278]:


iris["data"], iris['target']


# In[279]:


#Split the Dataset using DOB as random_state
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=705)


# In[280]:


print(len(X_train), len(X_test), len(y_train), len(y_test))

y_test


# In[281]:


new_train_data = []
for i in range(len(X_train)):
    print(X_train[i])
    print(y_train[i])
    new_train_data.append(np.append(X_train[i], y_train[i]).tolist())
    print(new_train_data)


# In[282]:


new_test_data = []
for i in range(len(X_test)):
    new_test_data.append(np.append(X_test[i], y_test[i]).tolist())
    print(X_test[i])
    print(y_test[i])
    print(new_test_data)
    


# In[283]:


#Used merge sort for sorting the distances & correspondig dataset
def mergeSort(nlist):
    #print("Splitting ",nlist)
    if len(nlist)>1:
        mid = len(nlist)//2
        lefthalf = nlist[:mid]
        righthalf = nlist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=j=k=0       
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i][1] < righthalf[j][1]:
                nlist[k]=lefthalf[i]
                i=i+1
            else:
                nlist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            nlist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            nlist[k]=righthalf[j]
            j=j+1
            k=k+1
            
        
#Finding the Euclidean distanct datapoints/vectors
def distance_Eucl(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance =distance + ((point1[i] - point2[i])**2)
    return sqrt(distance)
 
# Find neighbours
def neighbours_search(train_data, rows_test, knn_neighbours):
    distances = list()
    for rows_train in train_data:
        dist = distance_Eucl(rows_test, rows_train)
        distances.append((rows_train, dist))
    mergeSort(distances)    
    neighbors = list()
    for i in range(knn_neighbours):
        neighbors.append(distances[i][0])
    return neighbors
 
# Doing prediction with the neighbours
def classification_pred(train_data, rows_test, knn_neighbours):
    neighbors = neighbours_search(train_data, rows_test, knn_neighbours)
    op_val = [line[-1] for line in neighbors]
    prediction_val = max(set(op_val), key=op_val.count)
    return prediction_val
 
# KNN Algorithm
def KNN_Neighbors(train_data, test, knn_neighbours):
    preds = list()
    for row in test:
        output = classification_pred(train_data, row, knn_neighbours)
        preds.append(output)
#         print(predictions)
    return(preds)


# In[284]:


y_pred_data = KNN_Neighbors(new_train_data, new_test_data, 3)
y_pred = []

y_pred = y_pred_data
np.array(y_pred), y_test

# print(len(y_pred) , len(y_test))
# print(y_pred , y_test)
count=0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        count=count+1
# print(count)
Acc=count/len(y_pred)
print("Accuracy is :", Acc)
Err_Rate= 1-Acc
print("Error Rate is :",Err_Rate)


# # KNN ALgorithm for IONOSPHERE_DATASET(General K value(Commented) & 1-NN (using genfromtxt function)

# In[285]:


"""
Implemented the K Nearest Neighbours algorithm for a general K but used 1 & 3 instead of K in the programs
 
"""

from random import seed
from random import randrange
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
ion_data = np.genfromtxt("ionosphere.txt", delimiter=",")


# In[286]:


# print(ion_data)
# print(len(ion_data))
# len(ion_data[-1])
# x = ion_data[:, 0]
x = ion_data[:, :-1]
print(x[0])
print(x.shape)
y = ion_data[:, -1]
print(y.shape)


# In[287]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=705)


# In[288]:


# print(X_train, X_test, y_train, y_test)
print(X_train)
print(X_train.shape)


# In[289]:


new_train_data = []
for i in range(len(X_train)):
    new_train_data.append(np.append(X_train[i], y_train[i]).tolist())
print(new_train_data[262])
print(len(new_train_data))


# In[290]:


new_test_data = []
for i in range(len(X_test)):
    new_test_data.append(np.append(X_test[i], y_test[i]).tolist())
print(new_test_data[87])


# In[291]:



def mergeSort(nlist):
    #print("Splitting ",nlist)
    if len(nlist)>1:
        mid = len(nlist)//2
        lefthalf = nlist[:mid]
        righthalf = nlist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=j=k=0       
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i][1] < righthalf[j][1]:
                nlist[k]=lefthalf[i]
                i=i+1
            else:
                nlist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            nlist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            nlist[k]=righthalf[j]
            j=j+1
            k=k+1
            
        
#Finding the Euclidean distanct datapoints/vectors
def distance_Eucl(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance =distance + ((point1[i] - point2[i])**2)
    return sqrt(distance)
 
# Find neighbours
def neighbours_search(train_data, rows_test, knn_neighbours):
    distances = list()
    for rows_train in train_data:
        dist = distance_Eucl(rows_test, rows_train)
        distances.append((rows_train, dist))
    mergeSort(distances)    
    neighbors = list()
    for i in range(knn_neighbours):
        neighbors.append(distances[i][0])
    return neighbors
 
# Doing prediction with the neighbours
def classification_pred(train_data, rows_test, knn_neighbours):
    neighbors = neighbours_search(train_data, rows_test, knn_neighbours)
    op_val = [line[-1] for line in neighbors]
    prediction_val = max(set(op_val), key=op_val.count)
    return prediction_val
 
# KNN Algorithm
def KNN_Neighbors(train_data, test, knn_neighbours):
    preds = list()
    for row in test:
        output = classification_pred(train_data, row, knn_neighbours)
        preds.append(output)
#         print(predictions)
    return(preds)


# In[292]:


#for genralized k value we can use the below commented function:
# k=int(input("enter the k value of KNN"))
# y_pred_data = KNN_Neighbors(new_train_data, new_test_data, k)
y_pred_data = KNN_Neighbors(new_train_data, new_test_data, 1)
y_pred = []

y_pred = y_pred_data
np.array(y_pred), y_test

print(len(y_pred) , len(y_test))
print(y_pred , y_test)
count=0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        count=count+1
# print(count)
Acc=count/len(y_pred)
print("Accuracy is",Acc)
Err_Rate= 1-Acc
print("Error Rate is",Err_Rate)


# # IONOSPHERE_Dataset & 3-NN Algorithm (using genfromtxt function)

# In[293]:


"""
Implemented the K Nearest Neighbours algorithm for a general K but used 1 & 3 instead of K in the programs
 
"""

from random import seed
from random import randrange
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
ion_data = np.genfromtxt("ionosphere.txt", delimiter=",")


# In[294]:


x = ion_data[:, :-1]
print(x.shape)
y = ion_data[:, -1]
print(y.shape)


# In[295]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=705)
print(X_train)
print(X_train.shape)


# In[296]:


new_train_data = []
for i in range(len(X_train)):
    new_train_data.append(np.append(X_train[i], y_train[i]).tolist())
print(new_train_data[262])
print(len(new_train_data))


# In[297]:


new_test_data = []
for i in range(len(X_test)):
    new_test_data.append(np.append(X_test[i], y_test[i]).tolist())
print(new_test_data[87])


# In[298]:


def mergeSort(nlist):
    #print("Splitting ",nlist)
    if len(nlist)>1:
        mid = len(nlist)//2
        lefthalf = nlist[:mid]
        righthalf = nlist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=j=k=0       
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i][1] < righthalf[j][1]:
                nlist[k]=lefthalf[i]
                i=i+1
            else:
                nlist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            nlist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            nlist[k]=righthalf[j]
            j=j+1
            k=k+1
            
        
#Finding the Euclidean distanct datapoints/vectors
def distance_Eucl(point1, point2):
    distance = 0.0
    for i in range(len(point1)-1):
        distance =distance + ((point1[i] - point2[i])**2)
    return sqrt(distance)
 
# Find neighbours
def neighbours_search(train_data, rows_test, knn_neighbours):
    distances = list()
    for rows_train in train_data:
        dist = distance_Eucl(rows_test, rows_train)
        distances.append((rows_train, dist))
    mergeSort(distances)    
    neighbors = list()
    for i in range(knn_neighbours):
        neighbors.append(distances[i][0])
    return neighbors
 
# Doing prediction with the neighbours
def classification_pred(train_data, rows_test, knn_neighbours):
    neighbors = neighbours_search(train_data, rows_test, knn_neighbours)
    op_val = [line[-1] for line in neighbors]
    prediction_val = max(set(op_val), key=op_val.count)
    return prediction_val
 
# KNN Algorithm
def KNN_Neighbors(train_data, test, knn_neighbours):
    preds = list()
    for row in test:
        output = classification_pred(train_data, row, knn_neighbours)
        preds.append(output)
#         print(predictions)
    return(preds)


# In[299]:


y_pred_data = KNN_Neighbors(new_train_data, new_test_data, 3)
y_pred = []

y_pred = y_pred_data
np.array(y_pred), y_test

print(len(y_pred) , len(y_test))
print(y_pred , y_test)
count=0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        count=count+1
print(count)
Acc=count/len(y_pred)
print("Accuracy is",Acc)
Err_Rate= 1-Acc
print("Error Rate is",Err_Rate)


# In[ ]:




