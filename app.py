# main.py (FastAPI)
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, conint, constr
from predict import predict_score_and_proba

API_KEY = os.getenv("AI_API_KEY", "")
THRESH = float(os.getenv("AI_PASS_THRESHOLD", "0.6"))

class CandidateIn(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    interview_score: conint(ge=0, le=100)
    skill_score: conint(ge=0, le=100)
    personality_score: conint(ge=0, le=100)
    education_level: constr(strip_whitespace=True)
    recruitment_strategy: constr(strip_whitespace=True)
    experience_level: constr(strip_whitespace=True)
    status: constr(strip_whitespace=True)

class ScoreOut(BaseModel):
    ai_score: int
    ai_notes: str
    proba: float | None = None
    passed: bool

app = FastAPI(title="TemanHire AI Service")

@app.post("/score", response_model=ScoreOut)
def score(payload: CandidateIn, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

    ai_score, proba = predict_score_and_proba(payload.dict())
    # aturan pass:
    passed = False
    if proba is not None:
        passed = proba >= THRESH
    else:
        passed = ai_score >= int(THRESH * 100)

    notes = (f"Skor dihitung dari Interview={payload.interview_score}, "
             f"Skill={payload.skill_score}, Personality={payload.personality_score}, "
             f"Edu={payload.education_level}, Exp={payload.experience_level}.")
    return {"ai_score": ai_score, "ai_notes": notes, "proba": proba, "passed": passed}
