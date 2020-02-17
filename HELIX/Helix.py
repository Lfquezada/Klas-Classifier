
'''
H E L I X

a proyect by Luis Quezada
'''

print('\n\t\tH  E  L  I  X\n')

# Libraries
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Train all models and return the one with best accuracy
def getBestFitModel(X,y,scaled,testSize):
	# Vars
	allAlgorithmsNames = []
	allKs = [3,5,7,11,13,15]
	allKernelModes = ['linear','rbf','poly','sigmoid']
	classifiers = []
	allResults = []

	# Splitting into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

	# Feature Scaling
	if scaled:
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

	bestAccuracy = max(allResults)
	bestAccuracyIndex = allResults.index(bestAccuracy)
	bestFitAlgorithmName = allAlgorithmsNames[bestAccuracyIndex]
	bestFitClassifier = classifiers[bestAccuracyIndex]

	return bestFitClassifier,bestFitAlgorithmName,bestAccuracy


def showResults():
	# Showing results for all algorithms
	for i in range(0,len(allAlgorithmsNames)):
		print(str(allAlgorithmsNames[i]) + ': ' + str(allResults[i]))



'''
print('\n>< Predict')
inputData = [[5.5,2.6,4.4,1.2]]
#inputData = sc.transform(inputData)
	
allPredictions = []

for classifier in classifiers:
	allPredictions.append(classifier.predict(inputData))

finalPred = int(stats.mode(allPredictions)[0][0])

if finalPred == 1:
	print('Iris-setosa')
elif finalPred == 2:
	print('Iris-versicolor')
elif finalPred == 3:
	print('Iris-virginica')
'''






