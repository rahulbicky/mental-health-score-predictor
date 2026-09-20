import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

# ─── Robust model path ──────────────────────────────────────────────────────
# Uses the directory of this file so the path works whether you run uvicorn
# from the project root, a parent folder, or on Render.
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "Mental_Health_Model.pkl")

# Countries that have their own encoding bucket; everything else → "Other"
TOP_COUNTRIES = {"Other", "India", "USA", "Canada", "Australia", "UK",
                 "Germany", "Mexico", "Turkey", "France"}

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Score Predictor",
    description="Predicts a student's mental health score (0–10) from behavioural and lifestyle inputs.",
    version="1.0.0",
)

# Allow all origins so the frontend can call the API regardless of where it
# is hosted.  Tighten this to your Render frontend URL after deployment if
# you prefer stricter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ─────────────────────────────────────────────
class StudentData(BaseModel):
    age                     : int   = Field(..., ge=10, le=100)
    gender                  : Literal["Male", "Female"]
    country                 : str
    academic_level          : Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform      : Literal["Facebook", "LinkedIn", "Instagram", "Snapchat",
                                      "Twitter", "YouTube", "TikTok", "LINE",
                                      "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"]
    purpose_of_use          : Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal["Medium", "Low", "Very High", "High"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend entry point."""
    index = BASE_DIR / "index.html"
    return FileResponse(str(index))


@app.get("/health", tags=["General"])
def health():
    """Lightweight health-check endpoint for uptime monitors."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: StudentData):
    """
    Accepts a student's profile and returns a predicted mental health score
    between 0 (very poor) and 10 (excellent).
    """
    try:
        country_group = data.country if data.country in TOP_COUNTRIES else "Other"

        input_row = pd.DataFrame([{
            "Age"                    : data.age,
            "Gender"                 : data.gender,
            "Country"                : data.country,
            "Academic_Level"         : data.academic_level,
            "Most_Used_Platform"     : data.most_used_platform,
            "Purpose_Of_Use"         : data.purpose_of_use,
            "Avg_Daily_Usage_Hours"  : data.avg_daily_usage_hours,
            "Daily_Unlocks"          : data.daily_unlocks,
            "Study_Hours"            : data.study_hours,
            "Physical_Activity_Hours": data.physical_activity_hours,
            "Sleep_Hours_Per_Night"  : data.sleep_hours_per_night,
            "Stress_Level"           : data.stress_level,
            "Grouped_country"        : country_group,
        }])

        prediction = model.predict(input_row)[0]
        return PredictionResponse(
            predicted_mental_health_score=round(float(prediction), 2)
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


# ─── Serve static frontend (index.html / style.css / script.js) ─────────────
# This makes the frontend accessible at http://localhost:8000/ in the browser,
# with API calls going to the same origin — no CORS issues at all.
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    """Catch-all: serves index.html for any non-API path."""
    index = BASE_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "index.html not found"}