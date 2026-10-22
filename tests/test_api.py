from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

SAMPLE = {"Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
          "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal",
          "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat"}

def test_read_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_valid_probability():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200

def test_predict_rejects_bad_input():
    bad = {**SAMPLE, "Age": -5}
    assert client.post("/predict", json=bad).status_code == 422

def test_explain_returns_factors():
    r = client.post("/explain", json=SAMPLE)
    assert r.status_code == 200
    assert len(r.json()["Top 5 most impactful features"]) > 4