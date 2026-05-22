from fastapi import FastAPI
import random, socket, time, json, os, datetime

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/model")
LOG_DIR   = os.getenv("LOG_DIR", "/shared/logs")
APP_ENV   = os.getenv("APP_ENV", "dev")

def load_model_config():
    try:
        return json.load(open(f"{MODEL_DIR}/risk_model.json"))
    except:
        return {
            "model_version": "default",
            "threshold": 0.6,
            "accuracy": 0.90
        }

@app.get("/risk")
def score(message: str = ""):

    start = time.time()

    config = load_model_config()

    threshold = config.get("threshold", 0.6)

    msg = message.lower()

    score = 0.1

    high_risk_words = [
        "stolen",
        "fraud",
        "unauthorized",
        "hacked",
        "scam"
    ]

    medium_risk_words = [
        "lost",
        "blocked"
    ]

    for k in high_risk_words:
        if k in msg:
            score += 0.55

    for k in medium_risk_words:
        if k in msg:
            score += 0.25

    score += random.uniform(0.05, 0.15)

    score = round(min(score, 0.99), 2)

    flag = "high_risk" if score >= threshold else "low_risk"

    # logging
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        log = (
            f"{datetime.datetime.now()} "
            f"| score={score} "
            f"| flag={flag} "
            f"| message={message}\n"
        )

        open(f"{LOG_DIR}/risk.log", "a").write(log)
    except:
        pass

    return {
        "risk_score": score,
        "flag": flag,
        "model_version": config.get("model_version"),
        "model_accuracy": config.get("accuracy"),
        "processing_ms": round((time.time() - start) * 1000, 2),
        "served_by": socket.gethostname(),
        "environment": APP_ENV
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pod": socket.gethostname()
    }
