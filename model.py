import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import pickle
data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
                                                    


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()                                                

regressor.fit(X_train, y_train)

prediction = regressor.predict(X_test)


pickle.dump(regressor,open('model.pkl', 'wb'))

model= pickle.load(open('model.pkl', 'rb'))

print(regressor.predict([[6.5, 7, 9]]))