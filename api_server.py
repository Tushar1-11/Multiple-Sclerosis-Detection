"""
MSNet — Medical AI REST API & Dashboard Server
===============================================
FastAPI server serving real patient data from `train_data.csv`
and running live PyTorch 1D-CNN predictions with model explanation.

Usage:
  python api_server.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from models.tabular_msnet import TabularMSNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MSNet.API")

app = FastAPI(title="MSNet Medical AI API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "train_data.csv"
CHECKPOINT_PATH = BASE_DIR / "outputs" / "tabular_1dcnn" / "best_tabular_1dcnn.pt"
WEB_DIR = BASE_DIR / "web"

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSOR: Optional[TabularPreprocessor] = None
MODEL: Optional[TabularMSNet] = None
DF_RAW: Optional[pd.DataFrame] = None
X_PROCESSED: Optional[np.ndarray] = None
Y_TARGET: Optional[np.ndarray] = None

PHENOTYPE_LABELS = {0: "Healthy Control", 1: "Relapsing-Remitting (RRMS)", 2: "Progressive MS (SPMS/PPMS)"}
SEX_LABELS = {"0": "Female", "1": "Male", 0: "Female", 1: "Male"}


def get_edss_status(edss_val: Any) -> str:
    """Return descriptive EDSS disability tier."""
    if pd.isna(edss_val) or edss_val is None or str(edss_val).strip() == "":
        return "No EDSS Recorded (Healthy Control)"
    try:
        val = float(edss_val)
        if val == 0.0:
            return "EDSS 0.0 — Normal Neurological Exam"
        elif val <= 2.5:
            return f"EDSS {val:.1f} — Mild Disability"
        elif val <= 5.5:
            return f"EDSS {val:.1f} — Moderate Disability (Ambulatory)"
        else:
            return f"EDSS {val:.1f} — Severe Disability (Requires Assistance)"
    except Exception:
        return "Unknown EDSS"


def load_model_and_data():
    global PREPROCESSOR, MODEL, DF_RAW, X_PROCESSED, Y_TARGET

    logger.info("Loading model checkpoint and dataset...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_PATH}")

    # Load PyTorch checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    PREPROCESSOR = checkpoint["preprocessor"]

    # Load Raw CSV
    DF_RAW = PREPROCESSOR.load_data(DATA_PATH)
    X_PROCESSED, Y_TARGET = PREPROCESSOR.transform(DF_RAW)

    # Initialize PyTorch Model
    num_features = X_PROCESSED.shape[1]
    num_classes = len(np.unique(Y_TARGET))

    MODEL = TabularMSNet(
        in_features=num_features,
        num_classes=num_classes,
        embed_dim=256,
    ).to(DEVICE)

    MODEL.load_state_dict(checkpoint["model_state_dict"], strict=False)
    MODEL.eval()

    logger.info(f"Loaded {len(DF_RAW)} real patient records and PyTorch 1D-CNN model on {DEVICE}.")


# Run load on startup
@app.on_event("startup")
def startup_event():
    load_model_and_data()


# ----------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------

@app.get("/api/patients")
def get_patients(limit: int = 100, search: str = ""):
    """Return paginated list of real patients from train_data.csv."""
    if DF_RAW is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    patients = []
    for idx, row in DF_RAW.iterrows():
        patient_hash = str(row.get("Patient: Hash", f"patient_{idx}"))
        if search and search.lower() not in patient_hash.lower():
            continue

        edss_val = row.get("EDSS", None)
        phenotype_code = int(row.get("PHENOTYPE-CODE", 0))

        patients.append({
            "id": idx,
            "hash": patient_hash,
            "age": int(row.get("Age", 0)),
            "sex": SEX_LABELS.get(row.get("SEX-CODE"), "Unknown"),
            "group_code": int(row.get("GROUP-CODE", 0)),
            "phenotype_code": phenotype_code,
            "phenotype_label": PHENOTYPE_LABELS.get(phenotype_code, "Unknown"),
            "edss": float(edss_val) if pd.notna(edss_val) else None,
            "edss_status": get_edss_status(edss_val),
            "disease_duration": float(row.get("Disease duration", 0.0)) if pd.notna(row.get("Disease duration")) else 0.0,
            "t2_lv": float(row.get("T2 LV (cubic mm)", 0.0)) if pd.notna(row.get("T2 LV (cubic mm)")) else 0.0,
        })

        if len(patients) >= limit:
            break

    return {"total": len(DF_RAW), "count": len(patients), "patients": patients}


@app.get("/api/patient/{patient_id}")
def get_patient_detail(patient_id: int):
    """Return complete clinical and volumetric details for a single patient."""
    if DF_RAW is None or patient_id < 0 or patient_id >= len(DF_RAW):
        raise HTTPException(status_code=404, detail="Patient record not found")

    row = DF_RAW.iloc[patient_id]
    edss_val = row.get("EDSS", None)
    phenotype_code = int(row.get("PHENOTYPE-CODE", 0))

    volumetrics = {
        "ncorticalGREY": float(row.get("ncorticalGREY [ml]", 0.0)),
        "nWHITE": float(row.get("nWHITE [ml]", 0.0)),
        "nThalamus": float(row.get("nThalamus [ml]", 0.0)),
        "nHippocampus": float(row.get("nHippocampus [ml]", 0.0)),
        "nOtherDGM": float(row.get("nOtherDGM [ml]", 0.0)),
        "nBrainstem": float(row.get("nBrainstem [ml]", 0.0)),
        "nGM_Cerebellum": float(row.get("nGM_Cerebellum [ml]", 0.0)),
        "nWM_Cerebellum": float(row.get("nWM_Cerebellum [ml]", 0.0)),
    }

    return {
        "id": patient_id,
        "hash": str(row.get("Patient: Hash", f"patient_{patient_id}")),
        "age": int(row.get("Age", 0)),
        "sex": SEX_LABELS.get(row.get("SEX-CODE"), "Unknown"),
        "group_code": int(row.get("GROUP-CODE", 0)),
        "phenotype_code": phenotype_code,
        "phenotype_label": PHENOTYPE_LABELS.get(phenotype_code, "Unknown"),
        "institution_code": int(row.get("INSTITUTION-CODE", 1)),
        "edss": float(edss_val) if pd.notna(edss_val) else None,
        "edss_status": get_edss_status(edss_val),
        "disease_duration": float(row.get("Disease duration", 0.0)) if pd.notna(row.get("Disease duration")) else 0.0,
        "treatment_code": int(row.get("Treatment-code", 0)) if pd.notna(row.get("Treatment-code")) else None,
        "t2_lv": float(row.get("T2 LV (cubic mm)", 0.0)),
        "volumetrics": volumetrics,
    }


class PredictRequest(BaseModel):
    patient_id: Optional[int] = None
    age: Optional[float] = 45.0
    sex_code: Optional[int] = 1
    t2_lv: Optional[float] = 5000.0
    disease_duration: Optional[float] = 8.0
    edss: Optional[float] = 2.0
    treatment_code: Optional[int] = 1
    ncorticalGREY: Optional[float] = 600.0
    nWHITE: Optional[float] = 700.0
    nThalamus: Optional[float] = 20.0
    nHippocampus: Optional[float] = 10.0
    nOtherDGM: Optional[float] = 30.0
    nBrainstem: Optional[float] = 22.0
    nGM_Cerebellum: Optional[float] = 65.0
    nWM_Cerebellum: Optional[float] = 22.0


@app.post("/api/predict")
def predict_patient(req: PredictRequest):
    """Run live PyTorch 1D-CNN inference and return diagnostic subtype + feature importance."""
    if MODEL is None or PREPROCESSOR is None or DF_RAW is None:
        raise HTTPException(status_code=500, detail="Model engine not loaded")

    # If patient_id provided, use exact row from DF_RAW
    if req.patient_id is not None and 0 <= req.patient_id < len(DF_RAW):
        x_vec = X_PROCESSED[req.patient_id : req.patient_id + 1]
        raw_row = DF_RAW.iloc[req.patient_id]
        edss_val = raw_row.get("EDSS", None)
    else:
        # Build synthetic dataframe from request values
        sample_dict = {
            "Age": [req.age],
            "SEX-CODE": [req.sex_code],
            "GROUP-CODE": [1],
            "PHENOTYPE-CODE": [1],
            "INSTITUTION-CODE": [1],
            "T2 LV (cubic mm)": [req.t2_lv],
            "Disease duration": [req.disease_duration],
            "EDSS": [req.edss],
            "Treatment-code": [req.treatment_code],
            "ncorticalGREY [ml]": [req.ncorticalGREY],
            "nWHITE [ml]": [req.nWHITE],
            "nThalamus [ml]": [req.nThalamus],
            "nHippocampus [ml]": [req.nHippocampus],
            "nOtherDGM [ml]": [req.nOtherDGM],
            "nBrainstem [ml]": [req.nBrainstem],
            "nGM_Cerebellum [ml]": [req.nGM_Cerebellum],
            "nWM_Cerebellum [ml]": [req.nWM_Cerebellum],
        }
        df_sample = pd.DataFrame(sample_dict)
        x_vec, _ = PREPROCESSOR.transform(df_sample)
        edss_val = req.edss

    # Model inference
    x_tensor = torch.tensor(x_vec, dtype=torch.float32).to(DEVICE)
    x_tensor.requires_grad = True

    logits = MODEL(x_tensor)
    probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class] * 100)

    # Feature Importance Attribution (Gradient Saliency)
    top_logit = logits[0, pred_class]
    top_logit.backward()
    saliency = x_tensor.grad.abs().cpu().numpy()[0]

    feature_importances = []
    for f_name, imp in zip(PREPROCESSOR.feature_names, saliency):
        feature_importances.append({"feature": f_name, "saliency": float(imp)})

    feature_importances.sort(key=lambda item: item["saliency"], reverse=True)

    return {
        "predicted_class": pred_class,
        "predicted_label": PHENOTYPE_LABELS.get(pred_class, "Unknown"),
        "confidence": round(confidence, 2),
        "probabilities": {
            "Healthy Control": round(float(probs[0] * 100), 2),
            "RRMS": round(float(probs[1] * 100), 2),
            "Progressive MS": round(float(probs[2] * 100), 2),
        },
        "edss_status": get_edss_status(edss_val),
        "top_feature_importances": feature_importances[:8],
    }


# Mount Static Frontend
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def read_root():
    """Serve main web frontend HTML."""
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8080, reload=True)
