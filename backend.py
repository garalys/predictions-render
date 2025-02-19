from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import shutil

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load the saved Linear Regression model
model = joblib.load("model/linear_regression.pkl")

# ✅ Upload Excel File and Predict
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # ✅ Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Read Excel file
        df = pd.read_excel(file_path)

        # ✅ Ensure required columns exist
        required_columns = ['year','mileage','mpg']  # Adjust as needed
        if not all(col in df.columns for col in required_columns):
            return {"error": f"Excel file must contain the following columns: {required_columns}"}

        # ✅ Make Predictions for Each Row
        df["prediction"] = model.predict(df[required_columns])

        # ✅ Save Processed File
        result_path = f"processed_{file.filename}"
        df.to_excel(result_path, index=False)

        return {"message": "File processed successfully!", "download_url": f"https://your-render-app-url/download/{result_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
