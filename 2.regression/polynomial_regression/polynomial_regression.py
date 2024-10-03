# polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X,y)

# fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
# print(X_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


# visualizing the linear regression results

# plt.scatter(X,y,color="red")
# plt.plot(X,lin_reg.predict(X),color="blue")
# plt.title("Truth or Bluff (Linear Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# visualizing the polynomial regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

# plt.scatter(X,y,color="red")
# plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
# plt.title("Truth or Bluff (Polynomial Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")

# plt.show()


# predicting a new result with linear regression
y_pred=lin_reg.predict([[6.5]])
print(y_pred)


# predicting a new result with polynomial regression

y_pred2=lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(y_pred2)



