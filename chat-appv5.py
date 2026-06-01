from fastapi import FastAPI
import requests
import socket
import os
import json
import datetime

app = FastAPI()

BOT = os.getenv("BOT_NAME", "BankBot")
APP_ENV = os.getenv("APP_ENV", "production")

RISK_URL = os.getenv(
    "RISK_API_URL",
    "http://localhost:8001"
)

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/model"
)

LOG_DIR = os.getenv(
    "LOG_DIR",
    "/shared/logs"
)

MEMORY_FILE = f"{MODEL_DIR}/memory.json"
KB_FILE = f"{MODEL_DIR}/knowledge_base.json"
UNANSWERED_FILE = f"{MODEL_DIR}/unanswered_questions.json"

# --------------------------------------------------
# Create Files If Missing
# --------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

for file in [MEMORY_FILE, KB_FILE, UNANSWERED_FILE]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump([], f)

# --------------------------------------------------
# Load Memory
# --------------------------------------------------

try:
    conversation_memory = json.load(open(MEMORY_FILE))
except:
    conversation_memory = []

# --------------------------------------------------
# Knowledge Base Lookup
# --------------------------------------------------

def search_knowledge(question):

    try:

        kb = json.load(open(KB_FILE))

        for item in kb:

            if item["question"].lower().strip() == question.lower().strip():

                return item["answer"]

    except:
        pass

    return None

# --------------------------------------------------
# Save Unanswered Question
# --------------------------------------------------

def save_unanswered(question):

    try:
        questions = json.load(open(UNANSWERED_FILE))
    except:
        questions = []

    # prevent duplicates

    exists = False

    for item in questions:

        if item["question"].lower().strip() == question.lower().strip():

            exists = True
            break

    if not exists:

        questions.append({
            "question": question,
            "created_at": str(datetime.datetime.now())
        })

        with open(UNANSWERED_FILE, "w") as f:

            json.dump(
                questions,
                f,
                indent=2
            )

# --------------------------------------------------
# Intent Detection
# --------------------------------------------------

def detect_intent(message):

    msg = message.lower()

    if "balance" in msg:
        return "balance_check"

    if "card" in msg:
        return "card_issue"

    if "loan" in msg:
        return "loan_query"

    if "fraud" in msg:
        return "fraud_alert"

    if "stolen" in msg:
        return "fraud_alert"

    return "general_query"

# --------------------------------------------------
# Generate Reply
# --------------------------------------------------

def generate_reply(intent, risk_flag):

    if risk_flag == "high_risk":

        return (
            "High-risk activity detected. "
            "Your request has been escalated to the fraud team."
        )

    replies = {
        "balance_check":
            "Your balance request is being processed securely.",

        "card_issue":
            "We understand your card issue. Our support team is reviewing it.",

        "loan_query":
            "Loan support is available between 9 AM and 6 PM.",

        "fraud_alert":
            "Your fraud alert has been registered successfully.",

        "general_query":
            "I do not know the answer yet. The question has been queued for AI learning."
    }

    return replies.get(
        intent,
        "Support request received."
    )

# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.get("/chat")
def chat(message: str = "hello"):

    timestamp = str(datetime.datetime.now())

    # 1. Search Knowledge Base First

    kb_answer = search_knowledge(message)

    if kb_answer:

        return {
            "bot": BOT,
            "reply": kb_answer,
            "source": "knowledge_base",
            "served_by": socket.gethostname()
        }

    # 2. Save Unknown Question For CronJob

    save_unanswered(message)

    intent = detect_intent(message)

    # 3. Call Risk API

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

    except:

        risk = {
            "risk_score": "unavailable",
            "flag": "unknown"
        }

        model_ver = "unknown"
        model_acc = 0.0

    # 4. Generate Reply

    reply = generate_reply(
        intent,
        risk.get("flag")
    )

    # 5. Store Memory

    conversation_memory.append({
        "time": timestamp,
        "message": message,
        "intent": intent,
        "risk": risk.get("flag")
    })

    try:

        with open(MEMORY_FILE, "w") as f:

            json.dump(
                conversation_memory[-100:],
                f,
                indent=2
            )

    except:
        pass

    # 6. Logging

    try:

        os.makedirs(
            LOG_DIR,
            exist_ok=True
        )

        with open(
            f"{LOG_DIR}/chat.log",
            "a"
        ) as f:

            f.write(
                f"{timestamp} | "
                f"{message} | "
                f"{intent} | "
                f"{risk.get('flag')}\n"
            )

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
        "knowledge_found": False,
        "question_queued": True,
        "served_by": socket.gethostname()
    }

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "bot": BOT
    }

# --------------------------------------------------
# Ready
# --------------------------------------------------

@app.get("/ready")
def ready():

    try:

        r = requests.get(
            f"{RISK_URL}/health",
            timeout=2
        )

        if r.status_code == 200:

            return {
                "ready": True
            }

    except:
        pass

    return {
        "ready": False
    }

# --------------------------------------------------
# Memory
# --------------------------------------------------

@app.get("/memory")
def memory():

    return {
        "entries": len(
            conversation_memory
        ),
        "last_10":
            conversation_memory[-10:]
    }

# --------------------------------------------------
# Knowledge Base
# --------------------------------------------------

@app.get("/knowledge")
def knowledge():

    try:

        kb = json.load(open(KB_FILE))

        return {
            "entries": len(kb),
            "data": kb
        }

    except:

        return {
            "entries": 0
        }

# --------------------------------------------------
# Unanswered Questions
# --------------------------------------------------

@app.get("/unanswered")
def unanswered():

    try:

        questions = json.load(
            open(UNANSWERED_FILE)
        )

        return {
            "count": len(questions),
            "questions": questions
        }

    except:

        return {
            "count": 0
        }
