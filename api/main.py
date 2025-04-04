# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from core.predictor import predict_match
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # ⬅️ přidat
import numpy as np

app = FastAPI()

# 🌍 Přidej CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # můžeš později omezit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Datový model pro požadavek
class MatchRequest(BaseModel):
    league_code: str
    home_team: str
    away_team: str

# 🔮 Predikce přes endpoint
@app.post("/predict")
def predict(match: MatchRequest):
    result = predict_match(match.league_code, match.home_team, match.away_team)

    # Převod všech hodnot v dictu na standardní typy
    clean_result = {
        key: (
            float(val) if isinstance(val, (np.float32, np.float64)) else
            bool(val) if isinstance(val, (np.bool_)) else
            str(val) if isinstance(val, (np.str_)) else
            val
        )
        for key, val in result.items()
    }

    return JSONResponse(content=clean_result)
