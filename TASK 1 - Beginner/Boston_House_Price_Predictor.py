#Boston House price prediction: Using the provided dataset containing features such as a number of rooms, crime rates,
#and other relevant factors, design and implement a regression model to accurately predict Boston house prices. 
# Your solution should involve data preprocessing, model selection, training, and evaluation:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

data = pd.read_csv("C:\\Users\\vyomu\\OneDrive\\Desktop\\ShadowFox\\TASK 1 - Beginner\\HousingData.csv")

data.head(5)
print(data.isnull().sum())
data.info()

data['CRIM'] = data['CRIM'].fillna(data['CRIM'].mean())
data['ZN'] = data['ZN'].fillna(data['ZN'].mean())
data['INDUS'] = data['INDUS'].fillna(data['INDUS'].mean())
data['CHAS'] = data['CHAS'].fillna(data['CHAS'].mode()[0])
data['AGE'] = data['AGE'].fillna(data['AGE'].mean())
data['LSTAT'] = data['LSTAT'].fillna(data['LSTAT'].mean())

X = data.drop('MEDV', axis=1)
Y = data['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

prediction = model.predict(X_test)

R2_SCORE = r2_score(Y_test, prediction)
MAE = mean_absolute_error(Y_test, prediction)
RMSE = root_mean_squared_error(Y_test, prediction)

print(f"R2_SCORE : {R2_SCORE}\n")
print(f"MEAN_ABSOLUTE_ERROR : {MAE}\n")
print(f"ROOT_MEAN_SQUARED_ERROR : {RMSE}\n")

plt.scatter(Y_test, prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()