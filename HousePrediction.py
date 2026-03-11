import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

data=fetch_california_housing()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LinearRegression()
model2=RandomForestRegressor()
model2.fit(x_train,y_train)
model.fit(x_train,y_train)
prd2=model2.predict(x_test)
prd=model.predict(x_test)

e=mean_absolute_error(y_test,prd)
r2=r2_score(y_test,prd)
print("error:",e)
print("r2score:",r2)
e1=mean_absolute_error(y_test,prd2)
r22=r2_score(y_test,prd2)
print("error:",e1)
print("r2score:",r22)


plt.scatter(y_test,prd,color='red')
plt.scatter(y_test,prd2,color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
