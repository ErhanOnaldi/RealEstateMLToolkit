import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset into test and training sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training the model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifer = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifer.fit(x_train,y_train)

y_pred = classifer.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
print(cm)
print(accuracy_score(y_pred=y_pred,y_true=y_test))

