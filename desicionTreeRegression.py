import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("California_Houses.csv")
x = dataset.iloc[: ,1:].values
y = dataset.iloc[:,:1].values

from sklearn.model_selection import train_test_split 
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_true=y_test, y_pred= y_pred)
print(score)
