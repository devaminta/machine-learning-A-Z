# svr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values




''' from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)
# X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))


# fitting svr to the dataset
# create your regressor 

from sklearn.svm import SVR

regressor=SVR(kernel="rbf")
regressor.fit(X,y)


# predicting a new result with linear regression
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)


# visualizing the Svr regression results

plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



# visualizing the polynomial regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)