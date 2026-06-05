from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, json, socket, datetime, httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BOT           = os.getenv("BOT_NAME",    "BankBot")
APP_ENV       = os.getenv("APP_ENV",     "production")
DATA_DIR      = os.getenv("MODEL_DIR",   "/data")
OLLAMA_URL    = os.getenv("OLLAMA_URL",  "http://ollama-service:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL","llama3")

os.makedirs(DATA_DIR, exist_ok=True)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE   = os.path.join(BASE_DIR, "knowledge.json")
UNANSWERED_FILE  = os.path.join(DATA_DIR,  "unanswered.txt")

# ------------------------------------------------------------------
# Branch-only message — never resolved by bot or LLM
# ------------------------------------------------------------------

BRANCH_REPLY = (
    "⚠️ For security and compliance reasons, this matter requires "
    "in-person verification. Please visit your nearest branch with "
    "a valid photo ID. Our staff will assist you immediately."
)

# ------------------------------------------------------------------
# Init unanswered file
# ------------------------------------------------------------------

if not os.path.exists(UNANSWERED_FILE):
    open(UNANSWERED_FILE, "w", encoding="utf-8").close()

# ------------------------------------------------------------------
# Load knowledge base
# ------------------------------------------------------------------

try:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        KNOWLEDGE = json.load(f)
    print(f"Loaded {len(KNOWLEDGE)} knowledge entries")
except Exception as e:
    print("Knowledge load error:", e)
    KNOWLEDGE = []

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def find_entry(question: str):
    """Return the first matching knowledge entry or None."""
    q = question.lower().strip()
    for item in KNOWLEDGE:
        for kw in item.get("keywords", []):
            if kw.lower() in q:
                return item
    return None


def save_unanswered(question: str):
    try:
        with open(UNANSWERED_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now()} | {question}\n")
    except Exception as e:
        print("Unable to save unanswered question:", e)


async def ask_ollama(category: str, user_message: str) -> str:
    """
    Call Ollama for generic informational queries only.
    Strict system prompt: no sensitive ops, no personal data.
    """
    system_prompt = (
        "You are BankBot, a helpful banking information assistant. "
        "Answer ONLY generic banking education questions — definitions, "
        "how products work, general processes. "
        "DO NOT provide account-specific advice, personal financial advice, "
        "fraud handling, or anything requiring human intervention. "
        "Keep answers under 80 words. Be clear and professional."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload
            )
            data = resp.json()
            return data["message"]["content"].strip()
    except Exception as e:
        print("Ollama error:", e)
        return None          # fall back to rule answer

# ------------------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------------------

@app.get("/chat")
async def chat(message: str = ""):

    msg = message.lower().strip()

    # ── 1. Greetings ──────────────────────────────────────────────
    greetings = ["hi","hello","hey","good morning","good afternoon","good evening"]
    if msg in greetings:
        return {
            "bot":         BOT,
            "reply":       (
                "👋 I'm BankBot! I can help with general banking information. "
                "For sensitive matters like fraud or blocked accounts, "
                "please visit your nearest branch."
            ),
            "intent":      "greeting",
            "risk_score":  "N/A",
            "risk_flag":   "N/A",
            "served_by":   socket.gethostname(),
            "llm_used":    False,
            "llm_engine":  "N/A"
        }

    # ── 2. Knowledge base lookup ───────────────────────────────────
    entry = find_entry(msg)

    if entry:
        # ── 2a. Branch-only: never let bot/LLM answer ─────────────
        if entry.get("branch_only"):
            return {
                "bot":         BOT,
                "reply":       BRANCH_REPLY,
                "intent":      entry.get("category", "branch_only"),
                "risk_score":  "HIGH",
                "risk_flag":   "high_risk",
                "served_by":   socket.gethostname(),
                "llm_used":    False,
                "llm_engine":  "N/A"
            }

        # ── 2b. LLM-eligible: ask Ollama, fall back to rule ────────
        if entry.get("use_llm"):
            ollama_reply = await ask_ollama(
                entry.get("category", ""), message
            )
            if ollama_reply:
                return {
                    "bot":         BOT,
                    "reply":       ollama_reply,
                    "intent":      entry.get("category", "general"),
                    "risk_score":  "N/A",
                    "risk_flag":   "low_risk",
                    "served_by":   socket.gethostname(),
                    "llm_used":    True,
                    "llm_engine":  f"Ollama/{OLLAMA_MODEL}"
                }
            # Ollama unavailable — fall through to rule answer

        # ── 2c. Rule-based answer ──────────────────────────────────
        return {
            "bot":         BOT,
            "reply":       entry.get("answer", ""),
            "intent":      entry.get("category", "general"),
            "risk_score":  "N/A",
            "risk_flag":   "low_risk",
            "served_by":   socket.gethostname(),
            "llm_used":    False,
            "llm_engine":  "rule-engine"
        }

    # ── 3. Unknown — save and defer ────────────────────────────────
    save_unanswered(message)
    return {
        "bot":         BOT,
        "reply":       (
            "📝 Your question has been noted and saved. "
            "Our team will reach out to you shortly. "
            "For urgent matters, please visit your nearest branch."
        ),
        "intent":      "unanswered",
        "risk_score":  "N/A",
        "risk_flag":   "none",
        "served_by":   socket.gethostname(),
        "llm_used":    False,
        "llm_engine":  "N/A"
    }

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "bot": BOT, "env": APP_ENV}

# ------------------------------------------------------------------
# Knowledge
# ------------------------------------------------------------------

@app.get("/knowledge")
def knowledge():
    return {
        "entries":      len(KNOWLEDGE),
        "branch_only":  [i["category"] for i in KNOWLEDGE if i.get("branch_only")],
        "llm_enabled":  [i["category"] for i in KNOWLEDGE if i.get("use_llm")],
        "rule_based":   [i["category"] for i in KNOWLEDGE if not i.get("use_llm") and not i.get("branch_only")]
    }

# ------------------------------------------------------------------
# Unanswered
# ------------------------------------------------------------------

@app.get("/unanswered")
def unanswered():
    try:
        with open(UNANSWERED_FILE, "r", encoding="utf-8") as f:
            questions = f.readlines()
        return {"count": len(questions), "questions": questions[-100:]}
    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@app.get("/config")
def config():
    return {
        "bot":              BOT,
        "env":              APP_ENV,
        "data_dir":         DATA_DIR,
        "ollama_url":       OLLAMA_URL,
        "ollama_model":     OLLAMA_MODEL,
        "knowledge_file":   KNOWLEDGE_FILE,
        "unanswered_file":  UNANSWERED_FILE,
        "knowledge_entries": len(KNOWLEDGE)
    }

# ------------------------------------------------------------------
# Knowledge Upload Endpoint
# ------------------------------------------------------------------

@app.post("/knowledge/upload")
async def upload_knowledge(entries: list):
    """
    Replace the entire knowledge base at runtime.
    POST a JSON array of knowledge entries — same schema as knowledge.json.
    Changes persist in the container until restart (use a PVC mount for persistence).
    """
    global KNOWLEDGE
    try:
        # Validate minimum required fields
        for item in entries:
            if "category" not in item or "keywords" not in item or "answer" not in item:
                return {
                    "status": "error",
                    "message": "Each entry must have: category, keywords, answer"
                }
        KNOWLEDGE = entries
        # Persist to the knowledge file so restarts pick it up (if PVC mounted)
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(KNOWLEDGE, f, indent=2)
        return {
            "status":  "ok",
            "entries": len(KNOWLEDGE),
            "message": f"Knowledge base updated with {len(KNOWLEDGE)} entries"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/knowledge/list")
def knowledge_list():
    """Return full knowledge entries for the admin UI table."""
    return KNOWLEDGE


@app.post("/knowledge/add")
async def add_knowledge(entry: dict):
    """
    Add a single entry to the live knowledge base.
    Body: { "category": "...", "keywords": [...], "answer": "...", "use_llm": false, "branch_only": false }
    """
    global KNOWLEDGE
    try:
        if not all(k in entry for k in ["category", "keywords", "answer"]):
            return {"status": "error", "message": "Required: category, keywords, answer"}
        KNOWLEDGE.append(entry)
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(KNOWLEDGE, f, indent=2)
        return {
            "status":    "ok",
            "entries":   len(KNOWLEDGE),
            "added":     entry["category"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
