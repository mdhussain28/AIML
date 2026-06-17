from fastapi import FastAPI
import socket, os, json, datetime, random, pickle
import numpy as np

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
LOG_DIR   = os.getenv("LOG_DIR",   "/data/logs")
APP_ENV   = os.getenv("APP_ENV",   "dev")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

KEYWORDS = ["lost","stolen","fraud","unauthorized","blocked","compromised","suspicious","hacked"]

def extract_features(message):
    msg = message.lower()
    return [
        sum(1 for k in KEYWORDS if k in msg),
        min(len(msg) / 100, 1.0),
        1 if "card"    in msg else 0,
        1 if "account" in msg else 0,
        1 if any(w in msg for w in ["immediately","urgent","asap","now"]) else 0,
    ]

def load_model():
    try:
        with open(f"{MODEL_DIR}/risk_model.pkl", "rb") as f:
            model = pickle.load(f)
        meta = json.load(open(f"{MODEL_DIR}/risk_model.json"))
        return model, meta
    except:
        return None, {"model_version": "rule-based", "threshold": 0.6, "fit_status": "no model"}

@app.get("/risk")
def score(message: str = ""):
    model, meta = load_model()
    threshold   = meta.get("threshold", 0.6)

    if model:
        try:
            prob = model.predict_proba(np.array([extract_features(message)]))[0][1]
            sc   = round(float(prob), 2)
        except:
            sc = round(min(0.95, sum(0.2 for k in KEYWORDS if k in message.lower()) + random.uniform(0.05, 0.2)), 2)
    else:
        sc = round(min(0.95, sum(0.2 for k in KEYWORDS if k in message.lower()) + random.uniform(0.05, 0.2)), 2)

    flag = "high_risk" if sc > threshold else "low_risk"

    try:
        open(f"{LOG_DIR}/risk.log", "a").write(
            f"{datetime.datetime.now()} | score={sc} flag={flag} msg={message}\n")
    except:
        pass

    return {"risk_score": sc, "flag": flag,
            "model_version": meta.get("model_version", "unknown"),
            "served_by": socket.gethostname()}

@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "env": APP_ENV,
            "model_version": meta.get("model_version", "unknown"),
            "fit_status":    meta.get("fit_status",    "unknown")}

@app.get("/model-info")
def model_info():
    _, meta = load_model()
    return meta
