

# testing Klas Module
import pandas as pd
import matplotlib.pyplot as plt
from Klas import getBestFitModel,klasPredict

# Import dataset
dataset = pd.read_csv('iris-data-clean.csv')

dataset['class'] = dataset['class'].str.replace('Iris-setosa','1')
dataset['class'] = dataset['class'].str.replace('Iris-versicolor','2')
dataset['class'] = dataset['class'].str.replace('Iris-virginica','3')
dataset['class'] = dataset['class'].astype('float64')

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values

classifier,name,accuracy,allNames,allResults = getBestFitModel(X,y,scaled=True,testSize=0.25,returnAllResults=True)

print('\n>< Best fit algorithm: ' + name)
print('>< Accuracy: ' + str(accuracy*100)[:4] + '%')

# Showing results for all algorithms
plt.barh(allNames, allResults, align='center')
plt.xlabel('Accuracy')
plt.title('All results')
plt.show()

# Predictions
print('\n>< Prediction')

inputData = [[6.2,2.9,4.3,1.3]]

# Predict via the best fit model
finalPred = int(classifier.predict(inputData))

if finalPred == 1:
	print('Iris-setosa')
elif finalPred == 2:
	print('Iris-versicolor')
elif finalPred == 3:
	print('Iris-virginica')

# Predict via Helix.predict
finalPred = klasPredict(X,y,scaled=True,testSize=0.25,threshold=0.90,inputX=inputData)

if finalPred == 1:
	print('Iris-setosa')
elif finalPred == 2:
	print('Iris-versicolor')
elif finalPred == 3:
	print('Iris-virginica')
