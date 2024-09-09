import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np

HouseDF = pd.read_csv('USA_Housing.csv')

X = HouseDF[['Avg. Area Income']]
Y = HouseDF['Price']

plt.scatter(X, Y)
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.title('Avg. Area Income vs. House Price')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, Y)

HouseDF1 = pd.read_csv('example_housing_data.csv')
X_test_new = HouseDF1[['Avg. Area Income']]
Y_test_new = HouseDF1['Price']

X_test_new_scaled = scaler.transform(X_test_new)
predictions = model.predict(X_test_new_scaled)

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

income = float(input("Enter the average area income to predict house price: "))
scaled_income = scaler.transform([[income]])
predicted_price = model.predict(scaled_income)

print(f"The predicted house price for an average area income of {income} is: ${predicted_price[0]:,.2f}")
