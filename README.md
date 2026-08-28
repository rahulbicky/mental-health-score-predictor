# Mental Health Score Predictor

A machine-learning web application that predicts a student's mental health score (0–10) from behavioural, lifestyle, and demographic inputs.

**Stack:** Python · scikit-learn · FastAPI · Pydantic · HTML / CSS / JS

---

## How it works

```
User fills form → JavaScript fetch() → FastAPI /predict → ML pipeline → score returned → displayed in UI
```

---

## Run locally

### 1. Clone the repo
```bash
git clone https://github.com/rahulbicky/mental-health-score-predictor.git
cd mental-health-score-predictor
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the backend
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open the app
Visit **http://127.0.0.1:8000** in your browser.

The FastAPI backend also serves the frontend (`index.html`) at the root URL,  
so there is no separate server needed for the frontend.

### 6. Interactive API docs
Visit **http://127.0.0.1:8000/docs** (Swagger UI) to explore and test the API.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Confirm the API is running |
| `GET`  | `/health` | Lightweight health-check |
| `POST` | `/predict` | Return a predicted mental health score |

### POST /predict — request body

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 60,
  "study_hours": 5.0,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 7.0,
  "stress_level": "Medium"
}
```

### POST /predict — response

```json
{
  "predicted_mental_health_score": 6.42
}
```

---

## Deploy on Render

### Backend (Web Service)
| Setting | Value |
|---------|-------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Environment** | Python 3 |

### Frontend (Static Site)
After deploying the backend, update `API_BASE` in `script.js`:
```js
const API_BASE = "https://your-backend.onrender.com";
```
Then deploy the frontend as a **Render Static Site** pointing to the repo root.

---

## Project structure

```
mental-health-score-predictor/
├── main.py                  # FastAPI application
├── Mental_Health_Model.pkl  # Trained scikit-learn pipeline
├── index.html               # Frontend UI
├── script.js                # Frontend logic (fetch → /predict)
├── style.css                # Frontend styling
├── requirements.txt         # Python dependencies
├── ML_Project.ipynb         # Jupyter notebook (model training)
├── ML Project.html          # Build guide / reference doc
└── README.md
```

---

> ⚠️ This app is for informational and educational purposes only — it is not a clinical assessment.
