from fastapi import FastAPI
import random, socket, os, json, datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

app = FastAPI()

APP_ENV   = os.getenv("APP_ENV",   "dev")
LOG_DIR   = os.getenv("LOG_DIR",   "/data/logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

KEYWORDS = ["lost","stolen","fraud","unauthorized","blocked",
            "compromised","suspicious","hacked"]

def extract_features(message: str) -> list:
    msg      = message.lower()
    kw_count = sum(1 for k in KEYWORDS if k in msg)
    msg_len  = min(len(msg) / 100, 1.0)
    has_card = 1 if "card"    in msg else 0
    has_acc  = 1 if "account" in msg else 0
    urgency  = 1 if any(w in msg for w in ["immediately","urgent","asap","now"]) else 0
    return [kw_count, msg_len, has_card, has_acc, urgency]

def load_model():
    try:
        with open(f"{MODEL_DIR}/risk_model.pkl", "rb") as f:
            model = pickle.load(f)
        meta = json.load(open(f"{MODEL_DIR}/risk_model.json"))
        return model, meta
    except:
        return None, {
            "model_version": "default-rule-based",
            "threshold": 0.6,
            "keywords": KEYWORDS,
            "accuracy": 0.0,
            "fit_status": "no model trained"
        }

@app.get("/risk")
def score(message: str = ""):
    model, meta = load_model()
    threshold   = meta.get("threshold", 0.6)
    model_ver   = meta.get("model_version", "default")

    if model:
        features = extract_features(message)
        X        = np.array([features])
        try:
            prob = model.predict_proba(X)[0][1]
            sc   = round(float(prob), 2)
        except:
            sc = round(min(0.95, sum(0.2 for k in KEYWORDS if k in message.lower())
                          + random.uniform(0.05, 0.2)), 2)
    else:
        sc = round(min(0.95, sum(0.2 for k in KEYWORDS if k in message.lower())
                      + random.uniform(0.05, 0.2)), 2)

    flag = "high_risk" if sc > threshold else "low_risk"

    try:
        log = f"{datetime.datetime.now()} | risk | score={sc} flag={flag} model={model_ver} msg={message}\n"
        open(f"{LOG_DIR}/risk.log", "a").write(log)
    except:
        pass

    return {"risk_score": sc, "flag": flag,
            "model_version": model_ver, "env": APP_ENV,
            "served_by": socket.gethostname()}

@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "env": APP_ENV,
            "pod": socket.gethostname(),
            "fit_status": meta.get("fit_status", "unknown")}

@app.get("/model-info")
def model_info():
    _, meta = load_model()
    return meta
