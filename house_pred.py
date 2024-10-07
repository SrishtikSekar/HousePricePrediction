import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np

# Load the dataset
HouseDF = pd.read_csv('USA_Housing.csv')

# Select multiple features for the model
X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
             'Avg. Area Number of Bedrooms', 'Area Population']]
Y = HouseDF['Price']

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, Y_train)

# Make predictions using Linear Regression
linear_predictions = linear_model.predict(X_test_scaled)


# Evaluate Linear Regression
mae_linear = metrics.mean_absolute_error(Y_test, linear_predictions)
mse_linear = metrics.mean_squared_error(Y_test, linear_predictions)
rmse_linear = np.sqrt(mse_linear)
r2_linear = metrics.r2_score(Y_test, linear_predictions)

# Output the performance metrics
print('Linear Regression:')
print('Mean Absolute Error (MAE):', mae_linear)
print('Mean Squared Error (MSE):', mse_linear)
print('Root Mean Squared Error (RMSE):', rmse_linear)
print('R² Score:', r2_linear)

# Visualizing feature importance from Random Forest
plt.figure(figsize=(10, 6))
feature_importance = rf_model.feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Random Forest')
plt.show()

# Prediction based on user input
income = float(input("Enter the average area income to predict house price: "))
age = float(input("Enter the average house age to predict house price: "))
rooms = float(input("Enter the average number of rooms to predict house price: "))
bedrooms = float(input("Enter the average number of bedrooms to predict house price: "))
population = float(input("Enter the area population to predict house price: "))

user_input = np.array([[income, age, rooms, bedrooms, population]])
user_input_scaled = scaler.transform(user_input)
predicted_price = linear_model.predict(user_input_scaled)

print(f"The predicted house price for the given input is: ${predicted_price[0]:,.2f}")
