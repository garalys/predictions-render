from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import pandas as pd
import joblib
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware


# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the saved Linear Regression model
model = joblib.load("model/linear_regression.pkl")

# ✅ Upload Excel File and Predict
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # ✅ Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Read Excel file
        df = pd.read_excel(file_path)

        # ✅ Ensure required columns exist
        required_columns = ["year", "mileage", "mpg"]  # Adjust as needed
        if not all(col in df.columns for col in required_columns):
            return {"error": f"Excel file must contain the following columns: {required_columns}"}

        # ✅ Make Predictions for Each Row
        df["prediction"] = model.predict(df[required_columns])

        # ✅ Save Processed File
        result_filename = f"processed_{file.filename}"
        result_path = f"downloads/{result_filename}"

        # ✅ Ensure 'downloads' folder exists
        os.makedirs("downloads", exist_ok=True)

        df.to_excel(result_path, index=False)

        # ✅ Dynamically generate download URL based on Render URL
        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/download/{result_filename}"

        return {"message": "File processed successfully!", "download_url": download_url}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ Route to Serve Processed Files
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"downloads/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)