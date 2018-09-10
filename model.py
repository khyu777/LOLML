import pandas as pd
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

regular = pd.read_csv('Data/LCK_Summer_2018.csv', index_col=0)
playoff = pd.read_csv('Data/LCK Summer 2018 Playoffs.csv', index_col=0)

train_X = regular.iloc[:, 3:-1]
train_Y = regular.iloc[:, -1]
test_X = playoff.iloc[:, 3:-1]
test_Y = playoff.iloc[:, -1]

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