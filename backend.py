from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load the saved Linear Regression model
model = joblib.load("model/linear_regression.pkl")

# ✅ API Route for Predictions
@app.post("/predict/")
def predict(features: dict):
    try:
        # ✅ Convert input dictionary to DataFrame
        X_new = pd.DataFrame([features])

        # ✅ Make Prediction
        prediction = model.predict(X_new)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))