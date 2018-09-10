import pandas as pd
import os
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

filename = 'LCK Summer 2018'

regular = pd.read_csv('Data/' + filename + '.csv', index_col=0)
playoff = pd.read_csv('Data/' + filename + ' Playoffs.csv', index_col=0)

train_X = regular.iloc[:, 3:-1]
train_Y = regular.iloc[:, -1]
test_X = playoff.iloc[:, 3:-1]
test_Y = playoff.iloc[:, -1]
'''
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train_X)
train_X = pd.DataFrame(np_scaled)
print(train_X)

np_test_scaled = min_max_scaler.fit_transform(test_X)
test_X = pd.DataFrame(np_test_scaled)
print(test_X)
'''
lm = linear_model.LinearRegression()
lm.fit(train_X, train_Y)
y_pred = lm.predict(test_X)

# The coefficients
print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_Y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_Y, y_pred))

test_Y = pd.DataFrame(test_Y)
playoff['prediction'] = y_pred
print(playoff.iloc[:, 3:])
#playoff.to_csv('C:/Users/khyu7/Documents/Coding/Lol/Data/' + filename + '_pred.csv')