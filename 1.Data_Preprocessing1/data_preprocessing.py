# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values



# taking care of missing data
# from sklearn.preprocessing import Imputer

# imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
# print(X)




# encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# labelEncoder_X=LabelEncoder()
# X[:,0]=labelEncoder_X.fit_transform(X[:,0])


categorical_data=X[:,0].reshape(-1,1)
numerical_data=X[:,[1,2]]



oneHotEncoder=OneHotEncoder(sparse_output=False)
categorical_data=oneHotEncoder.fit_transform(categorical_data)

X_transform=np.hstack([categorical_data,numerical_data])
X=X_transform


labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)

# print(X)
# print("below is y")
# print(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))