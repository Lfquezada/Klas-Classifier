

# testing Helix Module
import pandas as pd
from Helix import getBestFitModel

# Import dataset
dataset = pd.read_csv('iris-data-clean.csv')

dataset['class'] = dataset['class'].str.replace('Iris-setosa','1')
dataset['class'] = dataset['class'].str.replace('Iris-versicolor','2')
dataset['class'] = dataset['class'].str.replace('Iris-virginica','3')
dataset['class'] = dataset['class'].astype('float64')

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values

classifier,name,accuracy = getBestFitModel(X,y,scaled=False,testSize=0.25)

print('\n>< Best fit algorithm: ' + name)
print('>< Accuracy: ' + str(accuracy*100)[:4] + '%')

print('\n>< Predict')
inputData = [[5.5,2.6,4.4,1.2]]
#inputData = sc.transform(inputData)

finalPred = int(classifier.predict(inputData))

if finalPred == 1:
	print('Iris-setosa')
elif finalPred == 2:
	print('Iris-versicolor')
elif finalPred == 3:
	print('Iris-virginica')
