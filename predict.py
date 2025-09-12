# predict.py — replika preprocessing Streamlit (1:1)
import numpy as np
import pandas as pd
import joblib

# ---- load artifacts ----
SCALER = joblib.load("scaler.pkl")
MODEL = joblib.load("model.pkl")

# ---- label set yg dipakai di Streamlit ----
STREAMLIT_EDU = ["Bachelor (S1)", "Bachelor (Univ Top Dunia ex:MIT,Oxford)", "Master", "PhD"]
STREAMLIT_RS  = ["Agresif (HR Ganas)", "Moderat", "Pasif (Stecu)"]
STREAMLIT_EXP = ["Junior", "Mid", "Senior"]

# ---- mapper fleksibel: terima enum dari DB-mu JUGA label Streamlit ----
def norm_edu(x: str) -> str:
    x = (x or "").strip()
    # enum DB -> label streamlit setara
    if x.upper() == "S1": return STREAMLIT_EDU[0]
    if x.upper() == "S2": return STREAMLIT_EDU[2]
    if x.upper() == "S3": return STREAMLIT_EDU[3]
    if x.upper() == "SMA":  # tidak ada di training → treat as S1 (paling mendekati/ netral)
        return STREAMLIT_EDU[0]
    # kalau sudah pakai label streamlit, kembalikan apa adanya
    if x in STREAMLIT_EDU: return x
    # fallback netral
    return STREAMLIT_EDU[0]

def norm_rs(x: str) -> str:
    x = (x or "").strip()
    if x.lower().startswith("agresif"): return STREAMLIT_RS[0]
    if x.lower().startswith("moderat"): return STREAMLIT_RS[1]
    if x.lower().startswith("pasif"):   return STREAMLIT_RS[2]
    if x in STREAMLIT_RS: return x
    return STREAMLIT_RS[1]

def norm_exp(x: str) -> str:
    x = (x or "").strip().capitalize()
    if x not in STREAMLIT_EXP: return "Junior"
    return x

# mapping integer seperti di Streamlit
edu_to_int = {
    STREAMLIT_EDU[0]: 1,  # Bachelor (S1)
    STREAMLIT_EDU[1]: 2,  # Bachelor Top Dunia
    STREAMLIT_EDU[2]: 3,  # Master
    STREAMLIT_EDU[3]: 4,  # PhD
}
rs_to_int = {
    STREAMLIT_RS[0]: 1,  # Agresif (HR Ganas)
    STREAMLIT_RS[1]: 2,  # Modorat
    STREAMLIT_RS[2]: 3,  # Pasif (Stecu)
}

FINAL_COLS = [
    "InterviewScore", "SkillScore", "PersonalityScore",
    "EducationLevel_2", "EducationLevel_3", "EducationLevel_4",
    "RecruitmentStrategy_2", "RecruitmentStrategy_3",
    "ExperienceLevel_Mid", "ExperienceLevel_Senior",
]

def build_features(
    interview_score: float,
    skill_score: float,
    personality_score: float,
    education_level: str,
    recruitment_strategy: str,
    experience_level: str,
) -> pd.DataFrame:
    # 1) scaling 3 skor (persis Streamlit)
    X_num = np.array([[interview_score, skill_score, personality_score]], dtype=float)
    scaled = SCALER.transform(X_num)  # -> [s_interview, s_skill, s_personality]
    s_interview, s_skill, s_personality = scaled[0, 0], scaled[0, 1], scaled[0, 2]

    # 2) OHE manual
    ohe_cols = {
        "EducationLevel_2": 0,
        "EducationLevel_3": 0,
        "EducationLevel_4": 0,
        "RecruitmentStrategy_2": 0,
        "RecruitmentStrategy_3": 0,
        "ExperienceLevel_Mid": 0,
        "ExperienceLevel_Senior": 0,
    }

    edu_label = norm_edu(education_level)
    edu_code = edu_to_int[edu_label]
    if edu_code >= 2:
        ohe_cols[f"EducationLevel_{edu_code}"] = 1

    rs_label = norm_rs(recruitment_strategy)
    rs_code = rs_to_int[rs_label]
    if rs_code >= 2:
        ohe_cols[f"RecruitmentStrategy_{rs_code}"] = 1

    exp_label = norm_exp(experience_level)
    if exp_label == "Mid":
        ohe_cols["ExperienceLevel_Mid"] = 1
    elif exp_label == "Senior":
        ohe_cols["ExperienceLevel_Senior"] = 1

    row = {
        "InterviewScore": s_interview,
        "SkillScore": s_skill,
        "PersonalityScore": s_personality,
        **ohe_cols
    }
    df = pd.DataFrame([row], columns=FINAL_COLS)
    return df

def predict_score_and_proba(payload: dict) -> tuple[int, float | None]:
    df_features = build_features(
        payload["interview_score"],
        payload["skill_score"],
        payload["personality_score"],
        payload["education_level"],
        payload["recruitment_strategy"],
        payload["experience_level"],
    )
    y_pred = MODEL.predict(df_features)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(df_features)[0][1])
    # convert proba ke 0..100 (jika ada)
    ai_score = int(round((proba * 100))) if proba is not None else int(y_pred) * 100
    return ai_score, proba
