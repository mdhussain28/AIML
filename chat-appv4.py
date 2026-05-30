from fastapi import FastAPI
import requests
import socket
import os
import json
import datetime

app = FastAPI()

PROMPT = os.getenv("SYSTEM_PROMPT", "You are a banking support assistant.")
RISK_URL = os.getenv("RISK_API_URL", "http://localhost:8001")
BOT = os.getenv("BOT_NAME", "BankBot")
APP_ENV = os.getenv("APP_ENV", "dev")
LOG_DIR = os.getenv("LOG_DIR", "/shared/logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/model")

# Load conversation memory from NFS
try:
    conversation_memory = json.load(
        open(f"{MODEL_DIR}/memory.json")
    )
    print(f"Loaded {len(conversation_memory)} memory entries from NFS")
except:
    conversation_memory = []
    print("No existing memory found - starting fresh")


def detect_intent(message: str):
    msg = message.lower()

    if "balance" in msg:
        return "balance_check"

    elif "card" in msg:
        return "card_issue"

    elif "loan" in msg:
        return "loan_query"

    elif "fraud" in msg:
        return "fraud_alert"

    elif "stolen" in msg:
        return "fraud_alert"

    elif "unauthorized" in msg:
        return "fraud_alert"

    elif "lost" in msg:
        return "card_issue"

    return "general_query"


def generate_reply(intent: str, risk_flag: str):

    if risk_flag == "high_risk":
        return "High-risk activity detected. Your request has been escalated to the fraud team."

    responses = {
        "balance_check": "Your balance request is being processed securely.",
        "card_issue": "We understand your card issue. Our support team is reviewing it.",
        "loan_query": "Loan support is available between 9 AM and 6 PM.",
        "fraud_alert": "Your fraud alert has been registered successfully.",
        "general_query": "Our support assistant is here to help you."
    }

    return responses.get(
        intent,
        "Support request received."
    )


@app.get("/chat")
def chat(message: str = "hello"):

    timestamp = str(datetime.datetime.now())
    intent = detect_intent(message)

    try:
        risk = requests.get(
            f"{RISK_URL}/risk",
            params={"message": message},
            timeout=3
        ).json()

        model_ver = risk.get(
            "model_version",
            "unknown"
        )

        model_acc = risk.get(
            "model_accuracy",
            0.0
        )

    except Exception:
        risk = {
            "risk_score": "unavailable",
            "flag": "unknown"
        }

        model_ver = "unavailable"
        model_acc = 0.0

    reply = generate_reply(
        intent,
        risk.get("flag", "unknown")
    )

    conversation_memory.append({
        "time": timestamp,
        "message": message,
        "intent": intent,
        "risk": risk.get("flag")
    })

    # Persist memory to NFS
    try:
        with open(
            f"{MODEL_DIR}/memory.json",
            "w"
        ) as f:
            json.dump(
                conversation_memory[-100:],
                f
            )
    except:
        pass

    # Write logs
    try:
        os.makedirs(
            LOG_DIR,
            exist_ok=True
        )

        log = (
            f"{timestamp} "
            f"| intent={intent} "
            f"| risk={risk.get('flag')} "
            f"| model={model_ver} "
            f"| message={message}\n"
        )

        open(
            f"{LOG_DIR}/chat.log",
            "a"
        ).write(log)

    except:
        pass

    return {
        "bot": BOT,
        "reply": reply,
        "intent": intent,
        "risk_score": risk.get("risk_score"),
        "risk_flag": risk.get("flag"),
        "model_version": model_ver,
        "model_accuracy": model_acc,
        "memory_size": len(conversation_memory),
        "environment": APP_ENV,
        "served_by": socket.gethostname()
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "bot": BOT,
        "env": APP_ENV
    }


@app.get("/ready")
def ready():

    try:
        r = requests.get(
            f"{RISK_URL}/ready",
            timeout=2
        )

        if r.status_code == 200:
            return {
                "ready": True,
                "risk_api": "reachable"
            }
    except:
        pass
    return {
        "ready": False,
        "risk_api": "unreachable"
    }
@app.get("/memory")
def memory():
    return {
        "total_conversations": len(conversation_memory),
        "last_10": conversation_memory[-10:],
        "memory_source": f"{MODEL_DIR}/memory.json"
    }
@app.get("/config")
def config():
    return {
        "bot": BOT,
        "env": APP_ENV,
        "prompt": PROMPT,
        "risk_url": RISK_URL,
        "log_dir": LOG_DIR,
        "model_dir": MODEL_DIR
    }
