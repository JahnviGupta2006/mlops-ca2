
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Find the model file in the same folder as main.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Load the trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Place model.joblib beside main.py."
    )

model = joblib.load(MODEL_PATH)

# Create the FastAPI application
app = FastAPI(
    title="Iris Flower Prediction API",
    description="Predict Iris flower species using a trained ML model.",
    version="1.0.0"
)


# Define the input data format
class IrisInput(BaseModel):
    SepalLengthCm: float = Field(gt=0)
    SepalWidthCm: float = Field(gt=0)
    PetalLengthCm: float = Field(gt=0)
    PetalWidthCm: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Iris Prediction API is running!",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: IrisInput):
    try:
        # Convert request data to a DataFrame
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        return {
            "predicted_species": str(prediction)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
