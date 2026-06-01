from fastapi import FastAPI
import socket
import time
import pickle
import os
import datetime

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/model")
LOG_DIR = os.getenv("LOG_DIR", "/shared/logs")
APP_ENV = os.getenv("APP_ENV", "dev")

MODEL_FILE = f"{MODEL_DIR}/risk_model.pkl"

# Load trained model from PVC/NFS
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    MODEL_STATUS = "loaded"
    MODEL_VERSION = "v3-trained"
    MODEL_ACCURACY = 0.95

    print(f"Model loaded successfully from {MODEL_FILE}")

except Exception as e:
    model = None

    MODEL_STATUS = f"error: {str(e)}"
    MODEL_VERSION = "unknown"
    MODEL_ACCURACY = 0.0

    print(f"Failed to load model: {e}")


@app.get("/risk")
def score(message: str = ""):

    start = time.time()

    if model is None:
        return {
            "risk_score": 0,
            "flag": "model_unavailable",
            "model_version": MODEL_VERSION,
            "model_accuracy": MODEL_ACCURACY,
            "model_status": MODEL_STATUS,
            "served_by": socket.gethostname()
        }

    probability = float(model.predict_proba([message])[0][1])

    prediction = int(model.predict([message])[0])

    flag = "high_risk" if prediction == 1 else "low_risk"

    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        log = (
            f"{datetime.datetime.now()} "
            f"| score={probability:.2f} "
            f"| flag={flag} "
            f"| message={message}\n"
        )

        open(f"{LOG_DIR}/risk.log", "a").write(log)

    except:
        pass

    return {
        "risk_score": round(probability, 2),
        "flag": flag,
        "model_version": MODEL_VERSION,
        "model_accuracy": MODEL_ACCURACY,
        "model_status": MODEL_STATUS,
        "processing_ms": round((time.time() - start) * 1000, 2),
        "served_by": socket.gethostname(),
        "environment": APP_ENV
    }


@app.get("/model-info")
def model_info():

    return {
        "model_file": MODEL_FILE,
        "model_version": MODEL_VERSION,
        "model_accuracy": MODEL_ACCURACY,
        "model_status": MODEL_STATUS,
        "exists": os.path.exists(MODEL_FILE)
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "pod": socket.gethostname()
    }


@app.get("/ready")
def ready():

    return {
        "ready": model is not None,
        "model_file": MODEL_FILE,
        "exists": os.path.exists(MODEL_FILE)
    }
