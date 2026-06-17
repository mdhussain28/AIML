from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests, socket, os, json, datetime, jwt
from passlib.context import CryptContext
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2, io
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Config from ConfigMap #1 (bankbot-env) ────────────────────────────────────
CHROMA_HOST     = os.getenv("CHROMA_HOST",      "chromadb-service")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT",  "8000"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS","3"))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",   "500"))
LOG_LEVEL       = os.getenv("LOG_LEVEL",        "info")
BOT_NAME        = os.getenv("BOT_NAME",         "BankBot")
APP_ENV         = os.getenv("APP_ENV",          "dev")
RISK_URL        = os.getenv("RISK_API_URL",     "http://risk-service:8001")
LOG_DIR         = os.getenv("LOG_DIR",          "/data/logs")
MODEL_DIR       = os.getenv("MODEL_DIR",        "/data/models")
JWT_ALGORITHM   = os.getenv("JWT_ALGORITHM",    "HS256")
JWT_EXPIRE      = int(os.getenv("JWT_EXPIRE",   "3600"))
# ── Secrets from bankbot-secret ───────────────────────────────────────────────
JWT_SECRET  = os.getenv("JWT_SECRET",  "changeme-secret-key")
USERS_JSON  = os.getenv("USERS_JSON",  '{"admin":"admin123","user":"user123"}')
USERS       = json.loads(USERS_JSON)
# ── ConfigMap #2 — RAG files mounted at /app/rag-data ────────────────────────
RAG_DATA_DIR   = "/app/rag-data"
SYSTEM_PROMPT  = "You are a banking assistant. Only answer using approved banking documents."
BANKING_POLICY = ""
FAQ_TEXT       = ""
try:
    with open(f"{RAG_DATA_DIR}/system_prompt.txt") as f:
        SYSTEM_PROMPT = f.read().strip()
    print(f"Loaded system_prompt.txt")
except Exception as e:
    print(f"system_prompt.txt not found, using default: {e}")
try:
    with open(f"{RAG_DATA_DIR}/banking_policy.txt") as f:
        BANKING_POLICY = f.read().strip()
    print(f"Loaded banking_policy.txt ({len(BANKING_POLICY)} chars)")
except Exception as e:
    print(f"banking_policy.txt not found: {e}")
try:
    with open(f"{RAG_DATA_DIR}/faq.txt") as f:
        FAQ_TEXT = f.read().strip()
    print(f"Loaded faq.txt ({len(FAQ_TEXT)} chars)")
except Exception as e:
    print(f"faq.txt not found: {e}")
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
# ── ChromaDB (0.4.22 fix: host/port only, no Settings()) ─────────────────────
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_client.heartbeat()
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    knowledge_col = chroma_client.get_or_create_collection(
        name="banking_knowledge",
        embedding_function=emb_fn
    )
    RAG_AVAILABLE = True
    doc_count = knowledge_col.count()
    print(f"ChromaDB connected: {CHROMA_HOST}:{CHROMA_PORT} | docs={doc_count}")
    # Auto-seed from mounted ConfigMap files if collection is empty
    if doc_count == 0 and (BANKING_POLICY or FAQ_TEXT):
        seed_text  = "\n\n".join(filter(None, [BANKING_POLICY, FAQ_TEXT]))
        chunks     = [seed_text[i:i+CHUNK_SIZE]
                      for i in range(0, len(seed_text), CHUNK_SIZE)
                      if seed_text[i:i+CHUNK_SIZE].strip()]
        ts  = datetime.datetime.now().timestamp()
        ids = [f"configmap_seed_{i}_{ts}" for i in range(len(chunks))]
        knowledge_col.add(documents=chunks, ids=ids)
        print(f"Auto-seeded {len(chunks)} chunks from ConfigMap files")
except Exception as e:
    RAG_AVAILABLE = False
    knowledge_col = None
    print(f"ChromaDB not available ({CHROMA_HOST}:{CHROMA_PORT}): {e}")
try:
    conversation_memory = json.load(open(f"{MODEL_DIR}/memory.json"))
    print(f"Loaded {len(conversation_memory)} memory entries")
except:
    conversation_memory = []
# ── Banking keywords ──────────────────────────────────────────────────────────
BANKING_KEYWORDS = [
    "account","balance","card","loan","transfer","payment","fraud","stolen",
    "lost","transaction","bank","credit","debit","atm","interest","mortgage",
    "deposit","withdrawal","unauthorized","blocked","statement","pin","limit",
    "fee","cheque","swift","iban","overdraft","savings","current","penalty",
    "closure","kyc","nominee","passbook","ifsc",
    "upi","neft","rtgs","imps","emi","fd","rd","nri","vpa","bhim",
    "netbanking","net banking","mobile banking","phonepe","gpay","paytm",
    "aadhaar","pan","cibil","rupay","locker","remittance","forex",
    "fixed deposit","recurring deposit","demand draft","noc","lien",
]
def is_banking_related(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in BANKING_KEYWORDS)
def detect_intent(msg: str) -> str:
    m = msg.lower()
    if "balance" in m:                                           return "balance_check"
    if "lost" in m and "card" in m:                              return "lost_card"
    if "card" in m:                                              return "card_issue"
    if "loan" in m or "mortgage" in m or "emi" in m:             return "loan_query"
    if "fraud" in m or "stolen" in m:                            return "fraud_alert"
    if "upi" in m or "neft" in m or "rtgs" in m or "imps" in m: return "payment_query"
    if "transfer" in m or "payment" in m:                        return "payment_query"
    if "unauthorized" in m:                                      return "fraud_alert"
    if "block" in m or "locked" in m:                            return "account_blocked"
    if "statement" in m:                                         return "statement_request"
    if "interest" in m or "rate" in m:                           return "interest_query"
    if "fd" in m or "fixed deposit" in m or "rd" in m:           return "transaction_query"
    if "deposit" in m or "withdrawal" in m:                      return "transaction_query"
    if "atm" in m:                                               return "atm_query"
    if "penalty" in m or "closure" in m:                         return "loan_closure"
    if "kyc" in m or "aadhaar" in m or "pan" in m:               return "kyc_query"
    if "savings" in m or "account" in m:                         return "account_info"
    return "general_banking"
def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRE),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload  = jwt.decode(credentials.credentials, JWT_SECRET,
                              algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
def rag_search(query: str) -> str:
    if not RAG_AVAILABLE or not knowledge_col:
        return ""
    try:
        count = knowledge_col.count()
        if count == 0:
            return ""
        results = knowledge_col.query(
            query_texts=[query],
            n_results=min(TOP_K_RESULTS, count)
        )
        docs = results.get("documents", [[]])[0]
        if docs:
            return (
                f"Retrieved {len(docs)} documents\n\n"
                + "\n\n".join([f"[Policy]: {d}" for d in docs])
            )
    except Exception as e:
        print(f"RAG search error: {e}")
    return ""
def fallback_reply(intent: str, risk_flag: str) -> str:
    if risk_flag == "high_risk":
        return " High-risk activity detected. Escalated to fraud team. Please call our 24/7 helpline immediately."
    responses = {
        "balance_check":     "Your balance request is being processed. Use our secure app or visit a branch.",
        "lost_card":         "We'll block your card immediately. Replacement arrives in 3-5 business days.",
        "card_issue":        "Our card support team is reviewing your issue. Resolution within 24 hours.",
        "loan_query":        "Loan advisors available Mon-Fri, 9AM-6PM. Shall I arrange a callback?",
        "fraud_alert":       "Fraud alert registered. Our security team will contact you within 1 hour.",
        "payment_query":     "UPI/NEFT/RTGS queries handled by 24/7 support. We'll review your transaction history.",
        "account_blocked":   "Account blocks reviewed immediately. Please verify your identity at any branch.",
        "statement_request": "Statements available in our mobile app or at any branch.",
        "interest_query":    "Savings: 3.5% p.a. | FD 1yr: 6.5% | Home loan: 8.5-10.5%. Speak to an advisor.",
        "transaction_query": "Recent transactions are viewable in our mobile app.",
        "atm_query":         "ATM locator available on our website and mobile app.",
        "loan_closure":      "Early closure penalty is 2% of outstanding principal (waived after 6 months for floating rate loans).",
        "kyc_query":         "KYC can be updated at any branch or via our mobile app using Aadhaar or PAN.",
        "account_info":      "A savings account earns interest on your deposits. Visit us or use our app to open one.",
        "general_banking":   "I'm BankBot, your assistant. I can help with accounts, cards, loans, UPI, NEFT, and more!",
    }
    return responses.get(intent, "Our support team will assist you shortly.")
# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/login")
def login(username: str, password: str):
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(username), "token_type": "bearer",
            "username": username, "expires_in": JWT_EXPIRE}
@app.get("/chat")
def chat(message: str = "hello", username: str = Depends(verify_token)):
    timestamp = str(datetime.datetime.now())
    intent    = detect_intent(message)
    if not is_banking_related(message):
        return {
            "bot": BOT_NAME, "reply": "I'm BankBot, your banking assistant. I can help with accounts, cards, loans, UPI, NEFT, and more!",
            "intent": "out_of_scope", "risk_score": 0, "risk_flag": "none",
            "model_version": "N/A", "memory_size": len(conversation_memory),
            "environment": APP_ENV, "served_by": socket.gethostname(),
            "rag_used": False, "user": username
        }
    model_ver = "not-loaded"
    try:
        meta      = json.load(open(f"{MODEL_DIR}/risk_model.json"))
        model_ver = meta.get("model_version", "unknown")
    except:
        pass
    try:
        r    = requests.get(f"{RISK_URL}/risk", params={"message": message}, timeout=3)
        risk = r.json()
        if "flag" not in risk:
            raise ValueError("bad response")
    except Exception as e:
        print(f"Risk API error: {e}")
        risk = {"risk_score": 0.1, "flag": "low_risk"}
    risk_flag   = risk.get("flag", "low_risk")
    rag_context = rag_search(message)
    rag_used    = bool(rag_context)
    # RAG result overrides fallback_reply when available
    if rag_used:
        reply = rag_context
    else:
        reply = fallback_reply(intent, risk_flag)
    conversation_memory.append({
        "time": timestamp, "user": username, "message": message,
        "intent": intent, "risk": risk_flag, "reply": reply[:100]
    })
    try:
        with open(f"{MODEL_DIR}/memory.json", "w") as f:
            json.dump(conversation_memory[-100:], f)
    except:
        pass
    try:
        log = f"{timestamp} | user={username} | intent={intent} | risk={risk_flag} | rag={rag_used} | msg={message}\n"
        open(f"{LOG_DIR}/chat.log", "a").write(log)
    except:
        pass
    return {
        "bot": BOT_NAME, "reply": reply, "intent": intent,
        "risk_score": risk.get("risk_score", 0), "risk_flag": risk_flag,
        "model_version": model_ver, "memory_size": len(conversation_memory),
        "environment": APP_ENV, "served_by": socket.gethostname(),
        "rag_used": rag_used, "user": username
    }
@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...),
                           username: str = Depends(verify_token)):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    content = await file.read()
    text    = ""
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    else:
        text = content.decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found in file")
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)
              if text[i:i+CHUNK_SIZE].strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated")
    ts  = datetime.datetime.now().timestamp()
    ids = [f"{file.filename}_{i}_{ts}" for i in range(len(chunks))]
    knowledge_col.add(documents=chunks, ids=ids)
    try:
        open(f"{LOG_DIR}/knowledge.log", "a").write(
            f"{datetime.datetime.now()} | UPLOAD | user={username} | file={file.filename} | chunks={len(chunks)}\n")
    except:
        pass
    return {"status": "uploaded", "file": file.filename,
            "chunks": len(chunks), "uploaded_by": username}
@app.get("/knowledge/search")
def search_knowledge(query: str, username: str = Depends(verify_token)):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    context = rag_search(query)
    return {"query": query, "results": context, "found": bool(context)}
@app.get("/knowledge/stats")
def knowledge_stats(username: str = Depends(verify_token)):
    if not RAG_AVAILABLE:
        return {"status": "ChromaDB not available", "total_documents": 0}
    try:
        return {"total_documents": knowledge_col.count(), "status": "ok",
                "top_k": TOP_K_RESULTS, "chunk_size": CHUNK_SIZE,
                "embedding_model": EMBEDDING_MODEL}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
@app.get("/knowledge/debug")
def knowledge_debug(username: str = Depends(verify_token)):
    info = {"rag_available": RAG_AVAILABLE, "chroma_host": CHROMA_HOST,
            "chroma_port": CHROMA_PORT, "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K_RESULTS, "chunk_size": CHUNK_SIZE,
            "rag_files_loaded": {
                "system_prompt": bool(SYSTEM_PROMPT),
                "banking_policy": bool(BANKING_POLICY),
                "faq": bool(FAQ_TEXT)
            }}
    if RAG_AVAILABLE:
        try:
            count = knowledge_col.count()
            info["doc_count"] = count
            if count > 0:
                info["sample_docs"] = knowledge_col.peek(limit=2).get("documents", [])
        except Exception as e:
            info["error"] = str(e)
    return info
@app.get("/health")
def health():
    return {"status": "ok", "bot": BOT_NAME, "env": APP_ENV,
            "rag_available": RAG_AVAILABLE, "served_by": socket.gethostname()}
@app.get("/ready")
def ready():
    try:
        r = requests.get(f"{RISK_URL}/health", timeout=2)
        if r.status_code == 200:
            return {"ready": True, "risk_api": "reachable"}
    except:
        pass
    return {"ready": False, "risk_api": "unreachable"}
@app.get("/memory")
def memory(username: str = Depends(verify_token)):
    user_mem = [m for m in conversation_memory if m.get("user") == username]
    return {"total": len(user_mem), "last_10": user_mem[-10:], "username": username}
@app.get("/config")
def config(username: str = Depends(verify_token)):
    return {"bot": BOT_NAME, "env": APP_ENV, "risk_url": RISK_URL,
            "rag_enabled": RAG_AVAILABLE, "chroma_host": CHROMA_HOST,
            "chroma_port": CHROMA_PORT, "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K_RESULTS, "chunk_size": CHUNK_SIZE,
            "log_dir": LOG_DIR, "model_dir": MODEL_DIR}
