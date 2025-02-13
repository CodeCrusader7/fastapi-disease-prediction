from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd

app = FastAPI()

# Get the absolute path of the current file
BASE_DIR = os.path.dirname(__file__)

# Define the model path
MODEL_PATH = os.path.join(BASE_DIR, "disease_prediction_model.pkl")

# Load the model
model = joblib.load(MODEL_PATH)

label_encoders = joblib.load(r"C:\Users\lenovo\Documents\FastAPI_Project\label_encoders.pkl")

@app.post("/predict/")
async def predict(data: dict):
    df = pd.DataFrame([data])

    # Encode categorical features
    categorical_cols = ["Animal_Type", "Breed", "Gender", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4",
                        "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing", "Lameness",
                        "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge"]

    for col in categorical_cols:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    # Convert to NumPy array and predict
    input_data = df.values.reshape(1, -1)
    prediction = model.predict(input_data)[0]

    # Convert prediction back to disease name
    predicted_disease = label_encoders["Disease_Prediction"].inverse_transform([prediction])[0]

    return {"predicted_disease": predicted_disease}
