from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from backend.ml_model import astronomy_ml

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

class PredictRequest(BaseModel):
    model: str = Field(default="rf", description="Model selector: 'rf' for Random Forest or 'xgb' for XGBoost")
    u: float = Field(..., ge=10.0, le=30.0, description="SDSS u-band magnitude (10 to 30)")
    g: float = Field(..., ge=10.0, le=30.0, description="SDSS g-band magnitude (10 to 30)")
    r: float = Field(..., ge=10.0, le=30.0, description="SDSS r-band magnitude (10 to 30)")
    i: float = Field(..., ge=10.0, le=30.0, description="SDSS i-band magnitude (10 to 30)")
    z: float = Field(..., ge=10.0, le=30.0, description="SDSS z-band magnitude (10 to 30)")
    redshift: float = Field(default=0.0, ge=-0.01, le=10.0, description="Spectroscopic redshift (-0.01 to 10.0)")

class ExplainRequest(BaseModel):
    model: str = Field(default="rf", description="Model selector: 'rf' for Random Forest or 'xgb' for XGBoost")
    u: float = Field(..., ge=10.0, le=30.0)
    g: float = Field(..., ge=10.0, le=30.0)
    r: float = Field(..., ge=10.0, le=30.0)
    i: float = Field(..., ge=10.0, le=30.0)
    z: float = Field(..., ge=10.0, le=30.0)
    redshift: float = Field(default=0.0, ge=-0.01, le=10.0)

class OrbitSimulateRequest(BaseModel):
    semi_major_axis_au: float = 1.0
    eccentricity: float = 0.0167
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    arg_per_deg: float = 0.0
    points: int = 100

class ClassifyLegacyRequest(BaseModel):
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: Optional[float] = 0.0

@router.get("/status")
def ml_status():
    """Return model and dataset readiness, active version, and available models."""
    return astronomy_ml.get_status()

@router.get("/dataset")
def ml_dataset(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(15, ge=1, le=100, description="Records per page"),
    class_filter: Optional[str] = Query(None, description="Optional class filter: STAR, GALAXY, or QUASAR")
):
    """Return dataset provenance metadata, missing-value summary, class counts, and paginated records."""
    return astronomy_ml.get_dataset_page(page=page, limit=limit, class_filter=class_filter)

@router.post("/train")
def ml_train():
    """Trigger verified model training pipeline for Random Forest and XGBoost."""
    try:
        from train_astronomy_models import train_and_explain
        train_and_explain()
        astronomy_ml._load_bundle()
        return {
            "status": "success",
            "message": "Random Forest and XGBoost models trained and validated with SHAP explainers.",
            "metrics": astronomy_ml.get_metrics_payload()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training job error: {str(e)}")

@router.get("/metrics")
def ml_metrics():
    """Return holdout accuracy, macro metrics, confusion matrices, and SHAP feature importance for RF & XGBoost."""
    return astronomy_ml.get_metrics_payload()

@router.post("/predict")
def ml_predict(req: PredictRequest):
    """Validate features and return model prediction and class probabilities."""
    return astronomy_ml.predict(
        model_key=req.model,
        u=req.u,
        g=req.g,
        r=req.r,
        i=req.i,
        z=req.z,
        redshift=req.redshift
    )

@router.post("/explain")
def ml_explain(req: ExplainRequest):
    """Compute local SHAP explanation for the input observation using the model's TreeExplainer."""
    return astronomy_ml.explain(
        model_key=req.model,
        u=req.u,
        g=req.g,
        r=req.r,
        i=req.i,
        z=req.z,
        redshift=req.redshift
    )

@router.post("/classify")
def ml_classify_legacy(req: ClassifyLegacyRequest):
    """Legacy classify route for backwards compatibility."""
    return astronomy_ml.classify(
        u=req.u,
        g=req.g,
        r=req.r,
        i=req.i,
        z=req.z,
        redshift=req.redshift or 0.0
    )

@router.post("/orbit-simulate")
def ml_orbit_simulate(req: OrbitSimulateRequest):
    """Keplerian orbit propagation service."""
    return astronomy_ml.simulate_orbit(
        a=req.semi_major_axis_au,
        e=req.eccentricity,
        inc_deg=req.inclination_deg,
        raan_deg=req.raan_deg,
        arg_per_deg=req.arg_per_deg,
        points=req.points
    )


class AnomalyRequest(BaseModel):
    observations: List[dict] = Field(..., min_length=1, max_length=200,
        description="List of observation dicts with keys: u, g, r, i, z, redshift")
    contamination: float = Field(default=0.1, ge=0.01, le=0.5,
        description="Expected fraction of anomalies (0.01 to 0.5)")


@router.post("/anomaly")
def ml_anomaly_detect(req: AnomalyRequest):
    """
    Unsupervised anomaly detection using Isolation Forest on SDSS-style photometric data.
    
    Returns anomaly scores and flags unusual observations. An 'anomaly' flag does NOT
    mean a confirmed unusual astronomical object — it indicates statistical deviation
    from the training distribution. Expert review is required.
    """
    import numpy as np
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise HTTPException(status_code=503, detail="scikit-learn not available for anomaly detection")

    features = ["u", "g", "r", "i", "z", "redshift"]
    rows = []
    for i, obs in enumerate(req.observations):
        try:
            row = [float(obs.get(f, 0.0)) for f in features]
            rows.append(row)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Observation {i} has invalid feature values")

    X = np.array(rows)

    # Fit Isolation Forest
    clf = IsolationForest(
        n_estimators=100,
        contamination=req.contamination,
        random_state=42
    )
    clf.fit(X)
    scores = clf.score_samples(X)       # negative anomaly scores (lower = more anomalous)
    labels = clf.predict(X)             # -1 = anomaly, 1 = normal

    results = []
    for i, (obs, score, label) in enumerate(zip(req.observations, scores, labels)):
        results.append({
            "index": i,
            "is_anomaly": bool(label == -1),
            "anomaly_score": round(float(score), 4),
            "observation": obs,
        })

    # Sort by most anomalous first
    results.sort(key=lambda x: x["anomaly_score"])

    return {
        "total": len(results),
        "anomaly_count": int(sum(1 for r in results if r["is_anomaly"])),
        "normal_count": int(sum(1 for r in results if not r["is_anomaly"])),
        "contamination_param": req.contamination,
        "results": results,
        "method": "Isolation Forest (unsupervised)",
        "features_used": features,
        "disclaimer": (
            "An anomaly flag indicates statistical deviation from the training distribution. "
            "It does NOT confirm an unusual astronomical object. "
            "Expert review and follow-up observation are required before any scientific conclusion."
        ),
        "model_info": {
            "algorithm": "Isolation Forest",
            "library": "scikit-learn",
            "n_estimators": 100,
            "random_seed": 42,
            "note": "Model fitted on the submitted observations only (unsupervised). No training data leakage."
        }
    }


@router.get("/model-card")
def ml_model_card():
    """
    Returns the model card for the SDSS galaxy/star/quasar classifier.
    Documents dataset provenance, training methodology, evaluation results, and limitations.
    """
    return {
        "model_name": "SDSS Photometric Classifier",
        "version": "2.0",
        "models": ["Random Forest Classifier", "XGBoost Classifier"],
        "task": "Multi-class classification: STAR / GALAXY / QUASAR",
        "dataset": {
            "name": "SDSS Sample Dataset (Synthetic SDSS-DR18-style)",
            "records": 1500,
            "class_balance": {"STAR": 500, "GALAXY": 500, "QUASAR": 500},
            "features": ["u", "g", "r", "i", "z", "redshift", "u-g", "g-r", "r-i", "i-z"],
            "provenance": "Generated to match SDSS DR18 photometric characteristics. Not the actual SDSS database.",
            "licence": "Educational use only"
        },
        "training": {
            "train_split": 0.7,
            "validation_split": 0.15,
            "test_split": 0.15,
            "stratified": True,
            "random_seed": 42,
            "preprocessing": "StandardScaler normalization; color index feature engineering"
        },
        "explainability": "SHAP TreeExplainer — global feature importance and per-record local explanations",
        "evaluation_note": "Holdout test set used for final metrics. No data leakage between train/test.",
        "limitations": [
            "Dataset is synthetic/sample; real SDSS data may show different distributions.",
            "Models may not generalize to data from other surveys (LSST, PanSTARRS, etc.)",
            "High redshift edge cases (z > 7) are underrepresented.",
            "This is an educational tool, not validated for scientific or operational decisions.",
        ],
        "intended_use": "Educational demonstration of astronomy ML workflows",
        "prohibited_use": "Scientific publications, operational classification, any safety-critical decision"
    }
