# multiple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# print(X)
# print(y)



# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X=LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
categorical_data=X[:,3].reshape(-1,1)
numerical_data=X[:,:-1]
oneHotEncoder=OneHotEncoder(sparse_output=False)
categorical_data=oneHotEncoder.fit_transform(categorical_data)
X_transform=np.hstack([categorical_data,numerical_data])
X=X_transform
# print(X)
# avoid the dummy variable trap
X=X[:,1:]
# print(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))'''



# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predicting the test set results
y_pred=regressor.predict(X_test)
# print(y_pred)


# Building the optimal model using backward elimination
import statsmodels.api  as sm

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
X_opt=X_opt.astype(float)
# print(X_opt.dtype)
# print(y.dtype)

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
summary=regressor_OLS.summary()
# print(summary)

X_opt=X[:,[0,1,3,4,5]]
X_opt=X_opt.astype(float)
# print(X_opt.dtype)
# print(y.dtype)

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
summary=regressor_OLS.summary()
# print(summary)


X_opt=X[:,[0,3,4,5]]
X_opt=X_opt.astype(float)
# print(X_opt.dtype)
# print(y.dtype)

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
summary=regressor_OLS.summary()
# print(summary)




X_opt=X[:,[0,3,5]]
X_opt=X_opt.astype(float)
# print(X_opt.dtype)
# print(y.dtype)

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
summary=regressor_OLS.summary()
# print(summary)



X_opt=X[:,[0,3]]
X_opt=X_opt.astype(float)
# print(X_opt.dtype)
# print(y.dtype)

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
summary=regressor_OLS.summary()
print(summary)