from fastapi import FastAPI
import random, socket, os, json, datetime

app = FastAPI()

APP_ENV   = os.getenv("APP_ENV",   "dev")
DB_USER   = os.getenv("DB_USERNAME", "not-set")
LOG_DIR   = os.getenv("LOG_DIR",   "/shared/logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/model")

def load_model_config():
    try:
        return json.load(open(f"{MODEL_DIR}/risk_model.json"))
    except:
        return {
            "model_version": "default",
            "threshold":     0.6,
            "keywords":      ["lost", "stolen", "fraud", "unauthorized", "blocked"],
            "accuracy":      0.0
        }

@app.get("/risk")
def score(message: str = ""):
    config    = load_model_config()
    keywords  = config.get("keywords",  [])
    threshold = config.get("threshold", 0.6)
    model_ver = config.get("model_version", "default")

    sc = round(
        min(0.95, sum(0.2 for k in keywords if k in message.lower()) + random.uniform(0.1, 0.3)),
        2
    )
    flag = "high_risk" if sc > threshold else "low_risk"

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log = f"{datetime.datetime.now()} | risk | score={sc} flag={flag} model={model_ver} msg={message}\n"
        open(f"{LOG_DIR}/risk.log", "a").write(log)
    except:
        pass

    return {
        "risk_score":    sc,
        "flag":          flag,
        "model_version": model_ver,
        "env":           APP_ENV,
        "served_by":     socket.gethostname()
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": APP_ENV, "pod": socket.gethostname()}

@app.get("/model-info")
def model_info():
    config = load_model_config()
    return {
        "model_version": config.get("model_version"),
        "threshold":     config.get("threshold"),
        "keywords":      config.get("keywords"),
        "accuracy":      config.get("accuracy"),
        "model_dir":     MODEL_DIR
    }
