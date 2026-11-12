"""Credit Scorecard API: calibrated LSTM credit scores with SHAP reason codes."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Paths resolve from this file, so the app starts from any working directory.
BASE_DIR = Path(__file__).resolve().parents[1]
ART = BASE_DIR / "artefacts"
SAMPLE_USERS = Path(os.getenv("SAMPLE_USERS_PATH", BASE_DIR / "api" / "sample_users.json"))

# ── Load artefacts ────────────────────────────────────────────────────────────

with open(ART / "meta.json") as f:
    meta = json.load(f)

cat_cols      = meta["cat_cols"]
num_cols      = meta["num_cols"]
cat_dims      = [tuple(x) for x in meta["cat_dims"]]
feature_names = cat_cols + num_cols          # full ordered list the model sees

# ── Loan-type column aggregation ──────────────────────────────────────────────
# TF-IDF expanded loan_type__ columns are kept separate for the model but
# collapsed back into a single "Loan_Type" token for SHAP display.
# If you used a different prefix, update this or add "tfidf_prefix" to meta.json.
TFIDF_PREFIX   = meta.get("tfidf_prefix", "loan_type__")
loan_type_cols = {f for f in feature_names if f.startswith(TFIDF_PREFIX)}

device = torch.device("cpu")   # CPU-only for serving

# ── Model classes (identical to notebook) ────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, cat_dims, num_cols, hidden=[128, 64, 32], dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sum(d for _, d in cat_dims) + len(num_cols),
            hidden_size=hidden[0], batch_first=True)
        self.embs = nn.ModuleList([nn.Embedding(v, d) for v, d in cat_dims])
        self.dropout = nn.Dropout(dropout)
        self.fc  = nn.Linear(hidden[0], hidden[1])
        self.net = nn.Sequential(
            self.dropout,
            self.fc,
            nn.BatchNorm1d(hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2]),
            nn.BatchNorm1d(hidden[2]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[2], 3),
        )

    def forward(self, cx, nx):
        x = torch.cat([e(cx[:, i]) for i, e in enumerate(self.embs)] + [nx], dim=1)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.net(lstm_out[:, -1, :])

class LSTMWrapper(nn.Module):
    """Single-tensor wrapper so GradientExplainer gets one input."""
    def __init__(self, lstm_clf, n_cat):
        super().__init__()
        self.model = lstm_clf
        self.n_cat = n_cat
    def forward(self, x):
        return self.model(x[:, :self.n_cat].long(), x[:, self.n_cat:].float())

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature

# ── Load weights ──────────────────────────────────────────────────────────────

lstm_clf = LSTMClassifier(cat_dims, num_cols)
# strict=True: fail loudly if the saved weights don't match the architecture.
lstm_clf.load_state_dict(torch.load(ART / "lstm_clf.pt", map_location=device), strict=True)
lstm_clf.eval()

temp_scaler = TemperatureScaler()
temp_scaler.load_state_dict(torch.load(ART / "temp_scaler.pt", map_location=device))
temp_scaler.eval()

scaler  = joblib.load(ART / "scaler.pkl")
le_dict = joblib.load(ART / "le_dict.pkl")
TEMPERATURE = float(temp_scaler.temperature.item())

# ── SHAP explainer ────────────────────────────────────────────────────────────

bg_np      = np.load(ART / "shap_background.npy")
background = torch.tensor(bg_np, dtype=torch.float32)
wrapped    = LSTMWrapper(lstm_clf, len(cat_cols))
explainer  = shap.GradientExplainer(wrapped, background)

# ── Risk score helpers ────────────────────────────────────────────────────────
#
# Points-to-double-odds (PDO) scaling, the industry standard for scorecards:
#   score = 600 + (50 / ln 2) × ln( P(Good) / P(Poor) )
# 600 points = even odds of Good vs Poor; every +50 points doubles those odds.
# Scores are clipped to 300–850.
#
PDO        = 50
BASE_SCORE = 600
FACTOR     = PDO / np.log(2)      # ≈ 72.13
EPSILON    = 1e-7                  # guard against ln(0)
 
# SHAP class weights in score-point space:
#   d(score)/d(logit_k) = [-FACTOR, 0, +FACTOR] for [Poor, Standard, Good]
#
# Standard's weight is exactly 0 — any shift in its logit moves P(Good) and
# P(Poor) proportionally, leaving their ratio (and the score) unchanged.
SHAP_SCORE_WEIGHTS = np.array([-FACTOR, 0.0, FACTOR])
 
def compute_risk_score(probs: np.ndarray) -> int:
    """probs: [p_poor, p_standard, p_good]"""
    odds  = (probs[2] + EPSILON) / (probs[0] + EPSILON)
    score = BASE_SCORE + FACTOR * np.log(odds)
    return int(round(np.clip(score, 300, 850)))
 
def score_to_band(s: int) -> str:
    if s >= 650: return "Excellent"   # odds ≥ 2
    if s >= 600: return "Good"        # 1 ≤ odds < 2
    if s >= 550: return "Fair"        # 0.5 ≤ odds < 1
    return "Poor"                     # odds < 0.5
 
def shap_to_score_points(shap_arr: np.ndarray) -> np.ndarray:
    """
    shap_arr: (1, n_features, 3) SHAP values of the RAW logits (GradientExplainer
    on the wrapped model, before temperature scaling).
    Returns:  (n_features,) — each feature's contribution in score points.

    ln(P(Good)/P(Poor)) = (logit_Good - logit_Poor) / T, so the score is linear
    in the logits and SHAP values map to points exactly: weights / T.
    """
    return shap_arr[0] @ (SHAP_SCORE_WEIGHTS / TEMPERATURE)
 
def aggregate_loan_type(sv_score: np.ndarray) -> dict:
    """
    Collapses all loan_type__ TF-IDF columns into a single 'Loan_Type' entry.
    Returns dict: {display_feature: score_points}
    """
    contribs = {}
    loan_total = 0.0
    for i, name in enumerate(feature_names):
        if name in loan_type_cols:
            loan_total += float(sv_score[i])
        else:
            contribs[name] = float(sv_score[i])
    if loan_type_cols:
        contribs["Loan_Type"] = loan_total
    return contribs
 
def top_factors(contribs: dict, n: int = 7):
    """Split contributions into positive/negative, top-n by absolute value."""
    ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    pos = [{"feature": k, "score_points": round(v, 1)} for k, v in ranked if v >= 0]
    neg = [{"feature": k, "score_points": round(v, 1)} for k, v in ranked if v <  0]
    return pos, neg


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Credit Scorecard API")

class UserFeatures(BaseModel):
    user_id: str
    features: dict

@app.get("/", response_class=HTMLResponse)
async def ui():
    return (BASE_DIR / "api" / "index.html").read_text()

@app.get("/users")
def list_users():
    if not SAMPLE_USERS.exists():
        return {"users": []}
    users = json.loads(SAMPLE_USERS.read_text())
    return {"users": [{"user_id": u["user_id"], "features": u["features"]} for u in users]}

@app.post("/score")
def score_user(payload: UserFeatures):
    # 1. Preprocess
    try:
        df = pd.DataFrame([payload.features])
        for col in cat_cols:
            df[col] = le_dict[col].transform(df[col].astype(str))
        df[num_cols] = scaler.transform(df[num_cols])
        cat_t = torch.tensor(df[cat_cols].values, dtype=torch.float32)
        num_t = torch.tensor(df[num_cols].values, dtype=torch.float32)
        x = torch.cat([cat_t, num_t], dim=1)   # (1, n_features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # 2. Calibrated probabilities
    with torch.no_grad():
        logits = lstm_clf(x[:, :len(cat_cols)].long(), x[:, len(cat_cols):].float())
        probs  = torch.softmax(temp_scaler(logits), dim=1).numpy()[0]

    # 3. Risk score (PDO log-odds scaling — see comments above)
    risk_score = compute_risk_score(probs)

    # 4. SHAP → score points
    #    GradientExplainer returns (1, n_features, n_classes)
    shap_arr  = np.array(explainer.shap_values(x))   # (1, n_features, 3)
    sv_score  = shap_to_score_points(shap_arr)        # (n_features,) in score-point space

    # 5. Collapse loan_type__ columns → single "Loan_Type"
    contribs = aggregate_loan_type(sv_score)

    # 6. Top 7 by absolute contribution
    pos_factors, neg_factors = top_factors(contribs, n=7)

    return {
        "user_id":    payload.user_id,
        "risk_score": risk_score,
        "risk_band":  score_to_band(risk_score),
        "probabilities": {
            "Poor":     round(float(probs[0]), 4),
            "Standard": round(float(probs[1]), 4),
            "Good":     round(float(probs[2]), 4),
        },
        "top_positive_factors": pos_factors,   # features that increased the score
        "top_negative_factors": neg_factors,   # features that decreased the score
    }

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)