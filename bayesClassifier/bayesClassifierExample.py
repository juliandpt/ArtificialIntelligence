# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:30:12 2019

@author: Esteban García Cuesta
"""
# Import packages
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# Import data
#training = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
#test = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')

training = pd.read_csv('C:\\Users\\5462\\Desktop\\tenis.csv')
test = pd.read_csv('C:\\Users\\5462\\Desktop\\tenisTest.csv')


# Create the X, Y, Training and Test
xtrain = training.drop('Play', axis=1)
ytrain = training.loc[:, 'Play']
xtest = test.drop('Play', axis=1)
ytest = test.loc[:, 'Play']

Outlook_enc = le.fit(xtrain.Outlook)
xtrain.Outlook = Outlook_enc.transform(xtrain.Outlook)

Temp_enc = le.fit(xtrain.Temp)
xtrain.Temp = Temp_enc.transform(xtrain.Temp)


Humidity_enc = le.fit(xtrain.Humidity)
xtrain.Humidity = Humidity_enc.transform(xtrain.Humidity)

Wind_enc = le.fit(xtrain.Wind)
xtrain.Wind = Wind_enc.transform(xtrain.Wind)
print(xtrain)

ytrain = le.fit_transform(ytrain)

Outlook_enc = le.fit(xtest.Outlook)
xtest.Outlook = Outlook_enc.transform(xtest.Outlook)

Temp_enc = le.fit(xtest.Temp)
xtest.Temp = Temp_enc.transform(xtest.Temp)


Humidity_enc = le.fit(xtest.Humidity)
xtest.Humidity = Humidity_enc.transform(xtest.Humidity)

Wind_enc = le.fit(xtest.Wind)
xtest.Wind = Wind_enc.transform(xtest.Wind)
print(xtest)
ytest = le.fit_transform(ytest)

# Init the Gaussian Classifier
model = GaussianNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output 
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
print("Pred",pred)
print("Real",ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')