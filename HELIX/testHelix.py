

# testing Helix Module
import pandas as pd
import matplotlib.pyplot as plt
from Helix import getBestFitModel,predict

# Import dataset
dataset = pd.read_csv('iris-data-clean.csv')

dataset['class'] = dataset['class'].str.replace('Iris-setosa','1')
dataset['class'] = dataset['class'].str.replace('Iris-versicolor','2')
dataset['class'] = dataset['class'].str.replace('Iris-virginica','3')
dataset['class'] = dataset['class'].astype('float64')

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values

classifier,name,accuracy,allNames,allResults = getBestFitModel(X,y,scaled=False,testSize=0.25,returnAllResults=True)

print('\n>< Best fit algorithm: ' + name)
print('>< Accuracy: ' + str(accuracy*100)[:4] + '%')

# Showing results for all algorithms
plt.barh(allNames, allResults, align='center')
plt.xlabel('Accuracy')
plt.title('All results')
plt.show()

# Predictions
print('\n>< Prediction')

inputData = [[5.5,2.6,4.4,1.2]]

# Predict via the best fit model
'''
#inputData = sc.transform(inputData)
finalPred = int(classifier.predict(inputData))
'''

# Predict via Helix.predict
finalPred = predict(X,y,scaled=False,testSize=0.25,threshold=0.8,inputX=inputData)


if finalPred == 1:
	print('Iris-setosa')
elif finalPred == 2:
	print('Iris-versicolor')
elif finalPred == 3:
	print('Iris-virginica')