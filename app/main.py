## main.py

from app.schemas import PatientIn
from fastapi import FastAPI
import joblib
import shap
import pandas as pd
from app.db import get_session, PredictionLog, create_db_and_tables, create_log, SessionDep, read_logs, PredictionLogPublic
from pathlib import Path
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title='Heart Disease API',lifespan=lifespan)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "pipeline.joblib"
model_pipe = joblib.load(MODEL_PATH)
model = model_pipe.named_steps['model']
preprocessor = model_pipe[:-1]
explainer = shap.TreeExplainer(model)
features = list(preprocessor['prep']['scalar'].get_feature_names_out())+list(preprocessor['prep']['encoder'].get_feature_names_out())


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p:PatientIn, session: SessionDep):
    patient = pd.DataFrame([p.model_dump()])
    prediction = float(model_pipe.predict_proba(patient)[0,1])

    log = PredictionLog(
        patientData = p.model_dump(),
        probability = prediction,
        label = int(prediction>0.5),
    )
    
    create_log(log,session)
    return {
        "Heart Disease": int(prediction>0.5),
        "Risk": f'{prediction*100:.2f}%'
    }
    
@app.post('/explain')
def explain(p:PatientIn):
    print(p.model_dump())
    patient = pd.DataFrame([p.model_dump()])
    X = preprocessor.transform(patient)
    shap_values = explainer.shap_values(X)[0]
    combined = sorted(zip(features,shap_values),key=lambda x:abs(x[1]),reverse=True)[:5]
    return {
        'score':round(float(sum(shap_values)),2),
        'Top 5 most impactful features':
        [
            {
            'feature':feat,
            'value':str(patient[feat.rsplit('_',maxsplit=1)[0]].values[0]),
            'influence':round(float(value),2),
            'Movement': 'Increases Risk' if (value>0) else 'Decreases Risk'
            }
        for feat,value in combined
        ]
    }

@app.get('/predictions', response_model=list[PredictionLogPublic])
def get_predictions(session: SessionDep):
    predictions = read_logs(session)
    return predictions
