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

#Training the machine
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score
score = r2_score(y_pred=y_pred,y_true=y_test)
print(score)