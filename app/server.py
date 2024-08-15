from fastapi import FastAPI
import joblib
import numpy as np

# Load your linear regression model
model = joblib.load('app/linear_regression_model.pkl')

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Linear Regression Model API'}

@app.post('/predict')
def predict(year: int):
    """
    Predicts the per capita income for a given year.

    Args:
        year (int): The year for which to predict the income.

    Returns:
        dict: A dictionary containing the predicted income.
    """        
    # Model expects a 2D array, hence reshape the input
    year_array = np.array([[year]])
    prediction = model.predict(year_array)
    predicted_income = prediction[0]
    return {'year': year, 'predicted_income': predicted_income}
