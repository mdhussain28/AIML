import numpy as np
import os, pickle, datetime, json, csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

MLFLOW_URL    = os.getenv("MLFLOW_URL",    "http://mlflow-service:5000")
MODEL_DIR     = os.getenv("MODEL_DIR",     "/data/models")
LOG_DIR       = os.getenv("LOG_DIR",       "/data/logs")
EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT", "bankbot-risk-model")
DATA_FILE     = os.getenv("TRAINING_DATA_FILE", "/app/training-data/data.csv")
NOISE_SEED    = int(os.getenv("NOISE_SEED",    "3"))
NOISE_SCALE   = float(os.getenv("NOISE_SCALE", "0.12"))
TEST_SIZE     = float(os.getenv("TEST_SIZE",   "0.25"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment(EXPERIMENT)

KEYWORDS = ["lost","stolen","fraud","unauthorized","blocked","compromised","suspicious","hacked"]

def extract_features(message: str) -> list:
    msg = message.lower()
    return [
        sum(1 for k in KEYWORDS if k in msg),
        min(len(msg) / 100, 1.0),
        1 if "card"    in msg else 0,
        1 if "account" in msg else 0,
        1 if any(w in msg for w in ["immediately","urgent","asap","now"]) else 0,
    ]

# ── Load training data from mounted ConfigMap file ───────────────────────────
def load_training_data(path):
    """
    Expects CSV with two columns: message,label
    label is 0 (low risk) or 1 (high risk)
    Lines starting with # are ignored (comments)
    """
    data = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            message, label = row[0].strip(), row[1].strip()
            if message.lower() == "message" and label.lower() == "label":
                continue  # skip header row
            try:
                data.append((message, int(label)))
            except ValueError:
                continue
    return data

print(f"Loading training data from: {DATA_FILE}")
training_data = load_training_data(DATA_FILE)

if len(training_data) < 10:
    raise RuntimeError(
        f"Only {len(training_data)} valid rows loaded from {DATA_FILE}. "
        f"Need at least 10 rows. Check ConfigMap formatting (message,label)."
    )

messages = [d[0] for d in training_data]
labels   = [d[1] for d in training_data]

print(f"Loaded {len(training_data)} rows "
      f"({sum(labels)} high-risk, {len(labels)-sum(labels)} low-risk)")

X = np.array([extract_features(m) for m in messages])
y = np.array(labels)

# Add Gaussian noise to numeric features to simulate real-world messiness.
# This is what makes regularization (C) actually produce different behavior —
# without noise, clean keyword-based data makes every C value converge identically.
rng   = np.random.default_rng(seed=NOISE_SEED)
noise = rng.normal(loc=0.0, scale=NOISE_SCALE, size=X.shape)
X     = X.astype(float) + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

print(f"MLflow: {MLFLOW_URL} | Experiment: {EXPERIMENT}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Three configs with meaningfully different regularization strength
configs = [
    {"C": 0.0008, "max_iter": 50,   "name": "underfit"},
    {"C": 8000,   "max_iter": 3000, "name": "overfit"},
    {"C": 0.8,    "max_iter": 300,  "name": "good_fit"},
]

best_model = None
best_meta  = {}

for cfg in configs:
    with mlflow.start_run(run_name=f"risk-{cfg['name']}"):

        mlflow.log_param("C",              cfg["C"])
        mlflow.log_param("max_iter",       cfg["max_iter"])
        mlflow.log_param("model",          cfg["name"])
        mlflow.log_param("data_file",      DATA_FILE)
        mlflow.log_param("training_rows",  len(training_data))
        mlflow.log_param("noise_seed",     NOISE_SEED)
        mlflow.log_param("noise_scale",    NOISE_SCALE)
        mlflow.set_tag("fit_type",         cfg["name"])

        model = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall    = recall_score(y_test,    test_pred, zero_division=0)
        f1        = f1_score(y_test,        test_pred, zero_division=0)
        gap       = train_acc - test_acc

        mlflow.log_metric("train_accuracy", round(train_acc, 3))
        mlflow.log_metric("test_accuracy",  round(test_acc,  3))
        mlflow.log_metric("precision",      round(precision,  3))
        mlflow.log_metric("recall",         round(recall,     3))
        mlflow.log_metric("f1_score",       round(f1,         3))
        mlflow.log_metric("overfit_gap",    round(gap,        3))

        mlflow.sklearn.log_model(model, artifact_path=cfg["name"])

        run_id = mlflow.active_run().info.run_id
        print(f"[{cfg['name']:>10}] C={cfg['C']:<8} train={train_acc:.3f} test={test_acc:.3f} "
              f"f1={f1:.3f} gap={gap:.3f} run_id={run_id}")

        if cfg["name"] == "good_fit":
            best_model = model
            best_meta  = {
                "model_version":  f"v-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "threshold":      0.5,
                "accuracy":       round(float(test_acc),  3),
                "f1_score":       round(float(f1),         3),
                "precision":      round(float(precision),  3),
                "recall":         round(float(recall),     3),
                "train_accuracy": round(float(train_acc),  3),
                "fit_status":     "good_fit",
                "mlflow_run_id":  run_id,
                "training_rows":  len(training_data),
                "trained_on":     str(datetime.datetime.now()),
            }

with open(f"{MODEL_DIR}/risk_model.pkl",  "wb") as f: pickle.dump(best_model, f)
with open(f"{MODEL_DIR}/risk_model.json", "w")  as f: json.dump(best_meta, f, indent=2)

print(f"\nBest model saved to {MODEL_DIR}")
