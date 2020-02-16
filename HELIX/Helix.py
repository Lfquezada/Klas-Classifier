
'''
H E L I X

a proyect by Luis Quezada
'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Vars
allAlgorithmsNames = []
allKs = [3,5,7,11]
allKernelModes = ['linear','rbf','poly','sigmoid']
classifiers = []
allResults = []

# Import dataset
dataset = pd.read_csv('iris-data-clean.csv')

dataset['class'] = dataset['class'].str.replace('Iris-setosa','1')
dataset['class'] = dataset['class'].str.replace('Iris-versicolor','2')
dataset['class'] = dataset['class'].str.replace('Iris-virginica','3')
dataset['class'] = dataset['class'].astype('float64')

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values

# Splitting into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision Tree Classification
allAlgorithmsNames.append('Decision Tree')
classifiers.append(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))

# Random Forest Classification
allAlgorithmsNames.append('Random Forest')
classifiers.append(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))

# SVM
for kernelMode in allKernelModes:
	allAlgorithmsNames.append('SVM ' + kernelMode)
	classifiers.append(SVC(kernel = kernelMode, random_state = 0))

# K-NN
for k in allKs:
	allAlgorithmsNames.append('K-NN k=' + str(k))
	classifiers.append(KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2))

# Naive Bayes
allAlgorithmsNames.append('Naive Bayes')
classifiers.append(GaussianNB())

# Logistic Regression
allAlgorithmsNames.append('Logistic Regression')
classifiers.append(LogisticRegression(random_state = 0))

# Fitting to all
for classifier in classifiers:
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	allResults.append(accuracy_score(y_test,y_pred))

# Showing results
for i in range(0,len(allAlgorithmsNames)):
	print(str(allAlgorithmsNames[i]) + ': ' + str(allResults[i]))










