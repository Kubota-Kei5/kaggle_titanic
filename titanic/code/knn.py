# %%
import pandas as pd
import matplotlib.pyplot as mb
import numpy as np

# %%
train=pd.read_csv('../Data/train.csv')
test=pd.read_csv('../Data/test.csv')
# %%
train.head()
# %%
train.dtypes
# %%
train.isnull().sum()
# %%
test.dtypes
# %%
test.isnull().sum()
# %%
x=train[['Pclass','SibSp','Parch']]
y=train[['Survived']]
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
# %%
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
# %%
knn.score(x_test, y_test)
# %%
x_for_submit = test[['Pclass', 'SibSp', 'Parch']]
submit = test[['PassengerId']]
submit['Survived']=knn.predict(x_for_submit)

submit
# %%
submit.to_csv('../submission/submit01_knn.csv', index=False)
# %%
