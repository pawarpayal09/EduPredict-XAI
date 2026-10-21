import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import FEATURES
from src.recommendations import build_action_plan
from src.service import EduPredictService, ValidationError

STUDENT = dict(gender="Female", study_time_hours=1.5, attendance_percent=68, sleep_hours=5.5,
               parental_education="High School", internet_access="No", extracurricular_activities="Yes",
               part_time_job="Yes", previous_grade=55)


@pytest.fixture(scope="module")
def svc():
    return EduPredictService()


def test_predict_one_fields(svc):
    r = svc.predict_one(STUDENT)
    assert 0 <= r["predicted_score"] <= 100
    assert r["interval_95"][0] <= r["interval_90"][0] <= r["interval_80"][0] <= r["predicted_score"] <= r["interval_80"][1] <= r["interval_90"][1] <= r["interval_95"][1]


def test_validation_rejects_bad_input(svc):
    with pytest.raises(ValidationError) as e:
        svc.validate({**STUDENT, "attendance_percent": 140, "gender": "x"})
    assert len(e.value.problems) == 2


def test_shap_is_additive(svc):
    row = svc.validate(STUDENT)
    base, sh = svc.explain(row)
    assert abs(base + sh.SHAP_Value.sum() - svc.model.predict(row)[0]) < 1e-3


def test_action_plan_never_uses_gender(svc):
    row = svc.validate(STUDENT)
    _, sh = svc.explain(row)
    plan = build_action_plan(row.iloc[0], sh, 60.0, svc)
    assert plan["actions"] and all(a["Feature"] != "gender" for a in plan["actions"])
    assert all(a["Est_gain"] >= 0.5 for a in plan["actions"])
    assert [a["Est_gain"] for a in plan["actions"]] == sorted((a["Est_gain"] for a in plan["actions"]), reverse=True)


def test_batch(svc):
    out = svc.batch(pd.DataFrame([STUDENT, STUDENT]))
    assert len(out) == 2 and {"Predicted_Score", "Lower_90", "Early_Warning", "Cohort_Band"} <= set(out.columns)


def test_api_endpoints():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert 0 <= c.post("/predict", json=STUDENT).json()["predicted_score"] <= 100
    r = c.post("/explain", json=STUDENT).json()
    assert len(r["shap"]) == len(FEATURES) and r["action_plan"]["actions"]
    assert c.post("/whatif", json=STUDENT).json()["plans"]
    assert c.post("/batch", json={"students": [STUDENT]}).json()["n"] == 1
    assert c.post("/predict", json={**STUDENT, "attendance_percent": 150}).status_code == 422
