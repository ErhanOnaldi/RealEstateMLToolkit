import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
#Importing the dataset
dataset = pd.read_csv("California_Houses.csv")
x = dataset.iloc[: ,1:].values
y = dataset.iloc[:,:1].values
#Splitting the dataset into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polyreg = PolynomialFeatures(degree = 2)
xpoly = polyreg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(xpoly,y_train)

y_pred = regressor.predict(polyreg.transform(x_test))

from sklearn.metrics import r2_score
score = r2_score(y_pred=y_pred,y_true=y_test)
print(score)