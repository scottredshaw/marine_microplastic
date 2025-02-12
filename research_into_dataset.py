# researching the dataset to see the best features to use to create valuable model

# goal - to determine and measure the increase in microplastics and what value of microplastic gievn
#  the date and the location

# exponential or linear regression
 
import pandas as pd
import numpy as np
import scipy
import matplotlib as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split


df_micro = pd.read_csv('ADVENTURE_MICRO_FROM_SCIENTIST.csv')

print(df_micro.head(5))

X = df_micro.iloc[:, [1,2]].to_numpy()

y = df_micro.iloc[:, 3].to_numpy()

std_scalar = preprocessing.StandardScaler()

X_std = std_scalar.fit_transform(X)

X_std[0].reshape(-1,1)

X_train, y_train, X_test, y_test = train_test_split(X_std, y, random_state=42)

regressor = linear_model.LinearRegression()

regressor.fit(X_train, y_train)

coef_1 = regressor.coef_

intercept = regressor.intercept_


# Plot line graph

plt.scatter(X_train, y_train, colour = "blue")

plt.plot(X_train, coef_1[0] * X_train + intercept, 'r')