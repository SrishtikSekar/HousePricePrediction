import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

try:
    HouseDF = pd.read_csv('USA_Housing.csv')
except FileNotFoundError:
    raise Exception("The file 'USA_Housing.csv' was not found. Please check the file path.")

X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
Y = HouseDF['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, Y_train)

try:
    HouseDF1 = pd.read_csv('example_housing_data.csv')
except FileNotFoundError:
    raise Exception("The file 'example_housing_data.csv' was not found. Please check the file path.")

X_test_new = HouseDF1[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                       'Avg. Area Number of Bedrooms', 'Area Population']]
Y_test_new = HouseDF1['Price']

X_test_new_scaled = scaler.transform(X_test_new)

predictions = lm.predict(X_test_new_scaled)

mae = metrics.mean_absolute_error(Y_test_new, predictions)
mse = metrics.mean_squared_error(Y_test_new, predictions)
rmse = np.sqrt(mse)
r2_score = metrics.r2_score(Y_test_new, predictions)

mean_price = Y_test_new.mean()
accuracy_percentage = (1 - mae / mean_price) * 100

print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('R^2 Score:', r2_score)
print('Accuracy (as percentage):', accuracy_percentage)
