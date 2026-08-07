from sklearn.datasets import fetch_california_housing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

california=fetch_california_housing()
print(california.DESCR)

dataset=pd.DataFrame(california.data,columns=california.feature_names)
dataset['Price']=california.target
print(dataset.head())

## Independent and Dependent features
X=dataset.iloc[:,:-1] #independent features
y=dataset.iloc[:,-1] #dependent features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()

## Prediction for the test data
y_pred=regression.predict(X_test)

## Performance Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

## R square and adjusted R square
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
print(1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

plt.scatter(y_test,y_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")
residuals=y_test-y_pred
sns.displot(residuals,kind="kde")

## SCatter plot with predictions and residual
##uniform distribution
plt.scatter(y_pred,residuals)

import pickle
pickle.dump(regression,open('regressor.pkl','wb'))
model=pickle.load(open('regressor.pkl','rb'))
model.predict(X_test)