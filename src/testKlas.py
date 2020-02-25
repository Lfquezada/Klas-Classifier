

# testing Klas Module
import pandas as pd
import matplotlib.pyplot as plt
from Klas import getBestFitModel,klasPredict

# Import dataset
dataset = pd.read_csv('iris-data-clean.csv')

flowerClasses = ['Iris-setosa','Iris-versicolor','Iris-virginica']

dataset['class'] = dataset['class'].str.replace(flowerClasses[0],'0')
dataset['class'] = dataset['class'].str.replace(flowerClasses[1],'1')
dataset['class'] = dataset['class'].str.replace(flowerClasses[2],'2')
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

# remove line 9 in cvs to test this input
#inputData = [[5.0,3.4,1.5,0.24999999999999997]]

# Predict via the best fit model
finalPred = int(classifier.predict(inputData))

# show prediction
print(flowerClasses[finalPred])

# Predict via Helix.predict
finalPred = klasPredict(X,y,scaled=True,testSize=0.25,threshold=0.90,inputX=inputData)

# show prediction
print(flowerClasses[finalPred])
