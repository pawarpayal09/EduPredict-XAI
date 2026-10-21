"""EduPredict-XAI REST API.   Run:  uvicorn api.main:app --reload     Docs: http://localhost:8000/docs"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommendations import build_action_plan
from src.service import EduPredictService, ValidationError

app = FastAPI(title="EduPredict-XAI API", version="3.0",
              description="Student exam-score prediction with uncertainty, SHAP explanations and action plans.")


@lru_cache(maxsize=1)
def svc() -> EduPredictService:
    return EduPredictService()


class Student(BaseModel):
    gender: Literal["Female", "Male"]
    study_time_hours: float = Field(ge=0, le=24)
    attendance_percent: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0, le=24)
    parental_education: Literal["High School", "Bachelors", "Masters", "PhD"]
    internet_access: Literal["Yes", "No"]
    extracurricular_activities: Literal["Yes", "No"]
    part_time_job: Literal["Yes", "No"]
    previous_grade: float = Field(ge=0, le=100)


class BatchRequest(BaseModel):
    students: list[Student] = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    s = svc()
    return {"status": "ok", "model": s.model_name, "note": s.note, "students_in_reference_data": len(s.df)}


@app.post("/predict")
def predict(student: Student):
    return svc().predict_one(student.model_dump())


@app.post("/explain")
def explain(student: Student):
    s = svc()
    try:
        row = s.validate(student.model_dump())
    except ValidationError as e:
        raise HTTPException(422, e.problems)
    pred = s.predict_one(student.model_dump())
    base, sh = s.explain(row)
    plan = build_action_plan(row.iloc[0], sh, pred["predicted_score"], s)
    return {"prediction": pred, "base_value": round(base, 2),
            "summary": s.explanation_text(pred["predicted_score"], base, sh),
            "shap": [{"feature": r.Feature, "value": str(r.Value), "shap_value": round(float(r.SHAP_Value), 3)}
                     for r in sh.itertuples()],
            "action_plan": {"actions": [{k: (v.item() if hasattr(v, "item") else v) for k, v in a.items()}
                                        for a in plan["actions"]],
                            "support": plan["support"], "strengths": plan["strengths"]}}


@app.post("/whatif")
def whatif(student: Student, target_gain: float = 5.0):
    s = svc()
    row = s.validate(student.model_dump()).iloc[0]
    base, plans, texts = s.whatif(row, target_gain)
    return {"baseline": round(base, 2), "target_gain": target_gain,
            "plans": [{"plan": t, "predicted_score": round(float(p.Predicted_score), 2), "gain": round(float(p.Gain), 2)}
                      for t, (_, p) in zip(texts, plans.iterrows())]}


@app.post("/batch")
def batch(req: BatchRequest):
    import pandas as pd
    out = svc().batch(pd.DataFrame([x.model_dump() for x in req.students]))
    return {"n": len(out), "early_warning_count": int(out.Early_Warning.sum()),
            "results": out.to_dict(orient="records")}
