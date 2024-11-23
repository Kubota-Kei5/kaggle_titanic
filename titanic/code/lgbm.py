#Light GBM（Light Gradient Boosting Model）
# %%
import pandas as pd
import matplotlib.pyplot as mb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
train=pd.read_csv('../Data/train.csv')
test=pd.read_csv('../Data/test.csv')
# %%
x=train[['Pclass','SibSp','Parch','Sex','Fare']]
y=train[['Survived']]

x=pd.get_dummies(x, dtype=int, columns=['Pclass', 'Sex'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
# %%
from sklearn.svm import LinearSVC

svc=LinearSVC()
svc.fit(x_train_scaled, y_train)

svc.score(x_test_scaled, y_test)
# %%
