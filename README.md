# Heart Risk API

**An explainable machine-learning service: from notebook to a tested, containerized, deployed REST API.**

![CI](https://github.com/Sujan-13/heart-risk-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-orange)
![SHAP](https://img.shields.io/badge/SHAP-explainability-blueviolet)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)

### 🔗 [Live demo (interactive API docs)](https://heart-risk-api-lda6.onrender.com/docs)

> The demo runs on a free tier, so the first request after inactivity can take about a minute while the service wakes up.
>
> **Disclaimer:** This is a learning project, not a medical device. Do not use it for medical decisions.

![Swagger UI of the live API](docs/swagger-ui.png)

---

## Why this project exists

Most machine-learning projects stop at a notebook. This one covers the rest of the path that a team needs before a model is useful:

1. **Model:** an XGBoost classifier in a scikit-learn pipeline, tuned with Optuna.
2. **Explainability:** every prediction can be explained with SHAP, so a user sees *why* the model answered as it did.
3. **API:** a validated REST service built with FastAPI.
4. **Persistence:** every prediction is logged to a database (PostgreSQL in production).
5. **Packaging:** a production-style Docker image (non-root user, health check).
6. **Quality:** automated tests with pytest, run on every push by GitHub Actions.
7. **Deployment:** live on Render, redeployed from the `main` branch.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predicted class and probability of heart disease |
| `/explain` | POST | The features that pushed this prediction up or down (SHAP values) |
| `/predictions` | GET | The 20 most recent logged predictions |
| `/health` | GET | Health check used by Docker and Render |

Inputs are validated with Pydantic. A bad value (for example a negative age) returns a `422` error with a clear message and never reaches the model.

### Try it

```bash
curl -X POST https://heart-risk-api-lda6.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
    "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat"
  }'
```

```json
{
  "Heart Disease": 0,
  "Risk": "5.88%"
}
```

`POST /explain` with the same body returns the strongest contributing features. A positive SHAP value pushes the prediction toward heart disease, and a negative value pushes it away:

```json
{
  "score": -2.98,
  "Top 5 most impactful features": [
    {
      "feature": "ST_Slope_Up",
      "value": "Up",
      "influence": -1.45,
      "Movement": "Decreases Risk"
    },
   ...
  ]
}
```

---

## Architecture

```mermaid
flowchart LR
    Client["Client or Swagger UI"] -->|"JSON over HTTPS"| API["FastAPI app in a Docker container"]
    API --> Val["Pydantic input validation"]
    Val --> Pipe["scikit-learn pipeline: KNN imputer, encoder, XGBoost"]
    Pipe --> API
    API -->|"POST /explain"| Shap["SHAP TreeExplainer"]
    API -->|"log every prediction"| DB[("PostgreSQL on Render")]
```

**Tech stack:** Python 3.12 · FastAPI · Pydantic · scikit-learn · XGBoost · Optuna · SHAP · SQLModel / SQLAlchemy · PostgreSQL · Docker · pytest · GitHub Actions · Render

---

## The model

- **Data:** [dataset name and link], [number of rows] patients, 11 input features, binary target `HeartDisease`.
- **Preprocessing (inside the pipeline):**
  - Zero values in `Cholesterol` are not real measurements, so they are treated as missing and imputed with KNN. Because this happens inside the pipeline, the imputer is fitted only on training data and nothing leaks from the test set.
  - Categorical features are one-hot encoded.
- **Model:** XGBoost, tuned with Optuna (100 trials) using cross-validation.
- **Explainability:** SHAP `TreeExplainer`.

### Results

| Metric | Value   |
|---|---------|
| Best cross-validation score (tuning) | ≈ 0.85  |

### What the model relies on

![SHAP summary plot](docs/shap_summary.png)

The most influential features are `ST_Slope`, `ChestPainType`, `Oldpeak`, `ExerciseAngina`, and `Sex`. This agrees with clinical knowledge: ST-segment slope, ST depression (`Oldpeak`), and exercise-induced angina from a stress test are core indicators of reduced blood flow to the heart. Seeing this agreement is a useful sanity check that the model learned real structure and not noise.

### Limitations

- The dataset is small and combines several sources, so results may not transfer to other populations.
- Patients with `ChestPainType = ASY` (no symptoms) are mostly disease cases in this data. This is most likely a sampling artifact and not a medical rule.
- The data contains many more men than women, so performance may differ by sex.
- SHAP shows what the model uses. It does not show what causes disease.

---

## Engineering

### Docker

The service is packaged with a production-style `Dockerfile`:

```dockerfile
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app/data

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt   # cached unless requirements change

COPY app ./app
COPY model ./model
RUN chown -R app:app /app
USER app                                              # never run as root

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`compose.yaml` runs the container for local use with a restart policy, the same health check, and a **named volume** (`api-data`) so the logged predictions survive container restarts.

### Database

Prediction logging uses SQLModel on top of SQLAlchemy. The connection string comes from the `DATABASE_URL` environment variable:

- **Production (Render):** a managed **PostgreSQL** database. The URL is stored as an environment variable on the host and is never committed to the repository.
- **Local and tests:** if `DATABASE_URL` is not set, the app falls back to a local SQLite file, so it runs and tests with no database server.

Tables are created on startup using FastAPI's `lifespan` handler.

### Testing

```bash
pytest
```

Tests use FastAPI's `TestClient` inside a pytest fixture, so the app's startup and shutdown (including table creation) run for each test:

```python
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
```

They cover:

- the health endpoint
- a valid prediction returns a probability between 0 and 1
- invalid input is rejected with a `422` error
- the explain endpoint returns contributing features

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`, so tests run the same way locally and in CI.

### Continuous integration (GitHub Actions)

```mermaid
flowchart LR
    A["git push or pull request"] --> B["GitHub Actions runner"]
    B --> C["Install dependencies"]
    C --> D["Run pytest"]
    D --> E["Build Docker image"]
    E --> F{"All steps pass?"}
    F -->|"yes"| G["Merge to main, Render redeploys"]
    F -->|"no"| H["Fix before merging"]
```

On every push and pull request, the workflow in `.github/workflows` installs the dependencies on a clean machine, runs the test suite, and builds the Docker image. A clean machine matters: it proves the project works from a fresh checkout and not only on my laptop. The badge at the top of this README shows the current status.

![GitHub Actions passing](docs/ci-passing.png)

### Deployment

The service runs on [Render](https://render.com) as a Docker web service built from this repository, with a managed PostgreSQL database. Pushing to `main` triggers a new build and deploy. Render uses `/health` to decide when the new version is ready.

---

## Run it yourself

### With Docker

```bash
git clone https://github.com/Sujan-13/heart-risk-api.git
cd heart-risk-api
docker compose up --build
```

Open http://localhost:8000/docs. Predictions are stored in SQLite inside a Docker volume.

### Without Docker

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on Linux or macOS)
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Use PostgreSQL locally

Create a `.env` file (it is in `.gitignore`) with one line and no quotes:

```
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### Configuration

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Database connection string | local SQLite file |

---

## Project structure

```
app/                 FastAPI application (main.py, schemas.py, db.py)
model/               Saved scikit-learn pipeline (pipeline.joblib)
tests/               API tests
Dataset/             Source data
train.ipynb          Model development, tuning, and SHAP analysis
Dockerfile           Production-style container image
compose.yaml         Local run with a persistent volume
pytest.ini           Test configuration
requirements.txt     Python dependencies
.github/workflows/   CI pipeline
```

## Retraining

Run `train.ipynb` to rebuild the pipeline and save it to `model/pipeline.joblib`. Use the library versions from `requirements.txt`, because a pipeline saved with one scikit-learn or XGBoost version may not load with another.

---

## Skills demonstrated

| Skill | Where to see it |
|---|---|
| Python backend and REST API design | `app/main.py`, `app/schemas.py` |
| Input validation | Pydantic models, `422` responses |
| ML pipelines and tuning | `train.ipynb` (scikit-learn, XGBoost, Optuna) |
| Explainable AI | SHAP plot and the `/explain` endpoint |
| Databases and ORM | `app/db.py` (SQLModel, PostgreSQL) |
| Docker | `Dockerfile`, `compose.yaml` |
| Automated testing | `tests/`, `pytest.ini` |
| CI/CD | `.github/workflows` |
| Cloud deployment | Render web service and managed database |
| Git workflow | Commit history and CI on every push |

## Possible improvements

- Add authentication and rate limiting
- Monitor the logged inputs for data drift
- Evaluate performance separately by sex and age group
- Validate on an external dataset

---

## Author

**Sujan Gupta** · Electronics and Communication Engineering graduate (2026), Thapathali Engineering Campus
Co-author of *Early Prediction of Chronic Kidney Disease Using Extra Trees and LightGBM with SHAP Visualization*, KEC Journal of Science and Engineering.

[GitHub](https://github.com/Sujan-13) · [LinkedIn](https://www.linkedin.com/in/sujan-g-545aa7187)