import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("California_Houses.csv")
x = dataset.iloc[: ,1:].values
y = dataset.iloc[:,:1].values

from sklearn.model_selection import train_test_split 
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
x_train = scX.fit_transform(x_train)
y_train = scY.fit_transform(y_train).ravel()

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train,y_train )

y_pred = scY.inverse_transform(regressor.predict(scX.transform(x_test)).reshape(-1, 1))


from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)