from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model (make sure the model file path is correct)
model = tf.keras.models.load_model('RNN_model_updated.keras')

# Define the data model for incoming ECG data
class ECGInput(BaseModel):
    data: list

# Root route to check if the API is running
@app.get("/")
def read_root():
    return {"message": "ECG Classification API is running!"}

# Endpoint for prediction
@app.post("/predict")
def predict(ecg_input: ECGInput):
    try:
        # Convert the input data to a NumPy array and ensure it's float
        input_data = np.array(ecg_input.data, dtype=float).reshape(1, -1)
        
        # Check if input_data has the correct shape (1, 187)
        if input_data.shape != (1, 187):
            raise ValueError("Input data must have exactly 187 values.")
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        # Handle errors and return a 400 HTTP status
        raise HTTPException(status_code=400, detail=str(e))
