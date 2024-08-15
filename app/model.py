# app/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np 


# Step 1: Read the CSV file
data = pd.read_csv('app/canada_per_capita_income.csv')

# Step 2: Data Preparation
X = data[['year']]
y = data['per capita income (US$)']

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Save the trained model to a file
model_filename = 'app/linear_regression_model.pkl'
joblib.dump(model, model_filename)

# Assuming 'model' is your trained Linear Regression model
year = np.array([[2000]])  # Reshape the input to match the expected shape
prediction = model.predict(year)
print(prediction)