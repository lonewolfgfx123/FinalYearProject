import pandas as pd
import numpy as np

from sklearn import linear_model

data = pd.read_csv('dataset.csv')
data.head()

X = data.iloc[:,3].values
y = data.iloc[:,4].values

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

score = lm.score(X,y)
print(score)
print(model.coef_)
