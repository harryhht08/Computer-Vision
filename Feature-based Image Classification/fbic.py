import os
import time
import cv2
from sklearn.cluster import KMeans
import numpy as np
# from PIL.Image import open    # This "open" might interfere with the python file I/O operations
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



# Define functions for later use
def calculateAccuracy(arr1, arr2):
    score = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            score += 1
    return score / len(arr1)



# 3.1 Find SIFT features

# 50 butterfly, with names starting with '024', class 0
# 50 cowboy, with names starting with '051', class 1
# 57 planes, with names starting with '251', class 2

dirs = os.listdir('Project2_data/TrainingDataset')

# Three arrays store descriptors separately
planes = []
cowboys = []
butterflies = []

ds = []
sift = cv2.SIFT_create()

for file in dirs:
    I = cv2.imread('Project2_data/TrainingDataset/' + file)
    # cv2.imshow('image', I)
    [f, d] = sift.detectAndCompute(I, None)
    if file.startswith('024'):
        butterflies.append(d)
    elif file.startswith('051'):
        cowboys.append(d)
    else:
        planes.append(d)

for b in butterflies:
    ds.append(b)
for c in cowboys:
    ds.append(c)
for p in planes:
    ds.append(p)
# The all descriptors list (ds) has the inside order of first 50 butterflies - then 50 cowboys - then 57 planes.


# 3.2, 3.3 Clustering & Histogram

# Merge all descriptors from all 157 images into all_d, of which we next perform k-means-clustering
all_d = []
for d in ds:
    for feature in d:
        all_d.append(feature)

kmeans = KMeans(n_clusters=100).fit(all_d)


# Loop over all 157 images and plot a histogram for each of them
# Rather than using kmeans.predict() for each feature, I use the computed labels

plotNumber = 1
all_centers = list(range(100))
all_centers_counts = []
for i in range(len(ds)):
    center_counts = [0] * 100
    for feature in ds[i]:
        # Notice! Make sure the input is 2D
        cluster = kmeans.predict([feature])
        cluster = cluster[0]
        center_counts[cluster] += 1

    all_centers_counts.append(center_counts)

    # Plot histograms of the 9 images, 3 of each type, so as to discover the pattern visually
    if plotNumber <= 3:
        plt.figure()
        plt.title('Histogram of Butterfly #' + str(plotNumber) + ' (not normalized)')
        plt.bar(all_centers, center_counts)
        plt.show()
    elif plotNumber >= 51 and plotNumber <= 53:
        plt.figure()
        plt.title('Histogram of Cowboy #' + str(plotNumber - 50) + ' (not normalized)')
        plt.bar(all_centers, center_counts)
        plt.show()
    elif plotNumber >= 101 and plotNumber <= 103:
        plt.figure()
        plt.title('Histogram of Plane #' + str(plotNumber - 100) + ' (not normalized)')
        plt.bar(all_centers, center_counts)
        plt.show()

    plotNumber += 1


# Normalize
normalized_centers_counts = []
for c in all_centers_counts:
    s = sum(c)
    newList = [x / s for x in c]
    normalized_centers_counts.append(newList)


# 3.4 Prepare for Classification - Repeat same operations on testing data

# While looping over all testing images, record the labels at the same time
# 10 b, 10 c, 16 p
planes = []
cowboys = []
butterflies = []

dirs = os.listdir('Project2_data/TestingDataset')
ds_testing = []
sift = cv2.SIFT_create()
for file in dirs:
    I = cv2.imread('Project2_data/TestingDataset/' + file)
    [f, d] = sift.detectAndCompute(I, None)
    if file.startswith('024'):
        butterflies.append(d)
    elif file.startswith('051'):
        cowboys.append(d)
    else:
        planes.append(d)

for b in butterflies:
    ds_testing.append(b)
for c in cowboys:
    ds_testing.append(c)
for p in planes:
    ds_testing.append(p)

all_d = []
for d in ds_testing:
    for feature in d:
        all_d.append(feature)

kmeans = KMeans(n_clusters=100).fit(all_d)
all_centers_counts = []
for i in range(len(ds_testing)):
    center_counts = [0] * 100
    for feature in ds_testing[i]:
        # Notice! Make sure the input is 2D
        cluster = kmeans.predict([feature])
        cluster = cluster[0]
        center_counts[cluster] += 1

    all_centers_counts.append(center_counts)

# Normalize
normalized_centers_counts_testing = []
for c in all_centers_counts:
    s = sum(c)
    newList = [x / s for x in c]
    normalized_centers_counts_testing.append(newList)


# 3.5 Classification
# 3.5.1 K=1 Nearest Neighbor

y_training = ['b'] * 50 + ['c'] * 50 + ['p'] * 57
X_training = normalized_centers_counts
y_testing = ['b'] * 10 + ['c'] * 10 + ['p'] * 16  # To be compared with the prediction results
X_testing = normalized_centers_counts_testing

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_training, y_training)
prediction = neigh.predict(X_testing)
accuracy_knn = calculateAccuracy(prediction, y_testing)
print(prediction)

# prediction = neigh.predict(X_training)
# accuracy = calculateAccuracy(prediction, y_training)


# 3.6 Linear SVM

clf = svm.SVC()
clf.fit(X_training, y_training)
prediction = clf.predict(X_testing)
accuracy_linearSVM = calculateAccuracy(prediction, y_testing)
print(prediction)

# 3.7 Kernel SVM
# Use the RBF function for the kernel

rbf = svm.SVC(kernel='rbf', random_state=1, gamma=0.008, C=0.1)
rbf.fit(X_training, y_training)
prediction = rbf.predict(X_testing)
accuracy_kernelSVM = calculateAccuracy(prediction, y_testing)
print(prediction)

print([accuracy_knn, accuracy_linearSVM, accuracy_kernelSVM])


print('checkPoint')
