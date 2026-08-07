import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LassoCV
from sklearn.metrics import mean_absolute_error,r2_score

dfs = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")
dfs = dfs.drop(columns=["day","month","year"])
dfs["Classes"] = np.where(dfs["Classes"].str.contains('not fire'),0,1)

#dependent and independent features
X = dfs.drop("FWI",axis=1)
y = dfs["FWI"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#independent and dependent features
#show map and find out the high correlation
'''
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)
plt.show()
'''

def find_correlation(data,value):
    corr_set = set()
    corr_values = data.corr()

    for i in range(len(corr_values.columns)):
        for j in range(i):
            if abs(corr_values.iloc[i,j]) > value:
                col_name = corr_values.columns[i]
                corr_set.add(col_name)
    return corr_set

#drop features for which threshhold is more than 0.85
drop_columns=find_correlation(X_train,0.85)

X_train.drop(drop_columns,axis=1,inplace=True)
X_test.drop(drop_columns,axis=1,inplace=True)

sd = StandardScaler()
X_train_scaled = sd.fit_transform(X_train)
X_test_scaled = sd.transform(X_test)

#Check boxplots to check standardization
'''
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(X_train_scaled)
plt.title("X plot before scaling")
plt.subplot(1,2,1)
sns.boxplot(X_test_scaled)
plt.title("X plot after scaling")
'''
regression = LinearRegression()
regression.fit(X_train_scaled,y_train)
y_pred = regression.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("mean absolute error",mae)
print("R2 score",score)

#plt.scatter(y_test,y_pred)
#plt.show()

#Lasso regression
lasso = Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("mean absolute error",mae)
print("R2 score",score)

#Ridge regression
ridge = Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("mean absolute error",mae)
print("R2 score",score)

#Elasti net regression
elsticnet = ElasticNet()
elsticnet.fit(X_train_scaled,y_train)
y_pred = elsticnet.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("mean absolute error",mae)
print("R2 score",score)

#Lasso CV regression
lassocv = LassoCV(cv=5)
lassocv.fit(X_train_scaled,y_train)
y_pred = elsticnet.predict(X_test_scaled)
print("alpha value",lassocv.alpha_)

#Model pickling
import pickle
pickle.dump(sd,open('scaler.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))
