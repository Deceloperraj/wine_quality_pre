import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('winequality.csv', sep=';')

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

sample_data = np.array([7.5, 0.3, 0.58, 1.8, 0.088, 16.0, 67.0, 0.9968, 3.31, 0.64, 9.5])
sample_data_df = pd.DataFrame([sample_data], columns=X.columns)

predicted_quality = model.predict(sample_data_df)
print(f"Predicted Wine Quality: {predicted_quality[0]:.2f}")

feature_names = X.columns
coefficients = model.coef_

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
