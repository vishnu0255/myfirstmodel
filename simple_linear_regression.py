import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Read the dataset
df=pd.read_csv('height-weight.csv')
df.head()

plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")

## divide our dataset into independent and dependent edatures
X=df[['Weight']] ##independent feature
y=df['Height'] ##dependent feature

## Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

## standardize the dataset Train independent data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
plt.scatter(X_train,y_train)

## Train the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("The slope or coefficient of weight is ",regressor.coef_)
print("Intercept:",regressor.intercept_)
plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),'r')
y_pred_test=regressor.predict(X_test)
plt.scatter(X_test,y_test)
plt.plot(X_test,regressor.predict(X_test),'r')

from sklearn.metrics import mean_squared_error,mean_absolute_error
mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred_test)
print(1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
scaled_weight=scaler.transform([[80]])
print("The height prediction for weight 80 kg is :",regressor.predict([scaled_weight[0]]))
## Assumptions
## plot a scatter plot for the prediction
plt.scatter(y_test,y_pred_test)

residuals=y_test-y_pred_test
## plot this residuals
import seaborn as sns
sns.distplot(residuals,kde=True)
## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(y_pred_test,residuals)