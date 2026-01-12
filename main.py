# app/main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List, Dict, Optional
import uuid
import time
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# -------------------------------
# CONFIG & LOGGER
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("zynexa")

# In-memory session store (later → Redis / DB)
sessions: Dict[str, List[Dict]] = {}

# -------------------------------
# MODELS
# -------------------------------
class Message(BaseModel):
    role: str          # "user" | "model"
    content: str
    timestamp: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    messages: List[Message]
    answer: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]
    created_at: float
    last_updated: float

# -------------------------------
# GEMINI SETUP
# -------------------------------
def get_gemini_model():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            system_instruction=(
                "You are Zynexa — helpful, concise, witty, and a bit sassy AI assistant. "
                "Answer naturally, use markdown when useful, be friendly but don't overdo emojis."
            )
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini → {e}")
        raise RuntimeError("Gemini initialization failed")

# -------------------------------
# SESSION MANAGEMENT
# -------------------------------
def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    sessions[session_id] = [
        {
            "role": "model",
            "content": "Hey there! I'm Zynexa — how can I make your day better? ✨",
            "timestamp": now
        }
    ]
    logger.info(f"New session created → {session_id}")
    return session_id

def get_or_create_session(session_id: Optional[str]) -> tuple[str, List[Dict]]:
    if not session_id or session_id not in sessions:
        session_id = create_new_session()
    return session_id, sessions[session_id]

# -------------------------------
# CHAT LOGIC
# -------------------------------
def generate_response(history: List[Dict], new_message: str, model) -> str:
    try:
        chat = model.start_chat(history=history[:-1])  # exclude last welcome if needed
        response = chat.send_message(new_message)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generation failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Sorry, Gemini is having a moment... Try again in a few seconds!"
        )

# -------------------------------
# LIFESPAN (startup / shutdown)
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ZYNEXA API is starting up... 🚀")
    yield
    logger.info("ZYNEXA API is shutting down... 👋")

# -------------------------------
# APP
# -------------------------------
app = FastAPI(
    title="ZYNEXA Chat API",
    description="Fast & simple multi-turn Gemini chat API",
    version="2025.1",
    lifespan=lifespan
)

# CORS - adjust origins in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# ENDPOINTS
# -------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()

    session_id, history = get_or_create_session(request.session_id)

    # Add user message
    user_msg = {
        "role": "user",
        "content": request.message,
        "timestamp": time.time()
    }
    history.append(user_msg)

    # Generate
    model = get_gemini_model()
    answer = generate_response(history, request.message, model)

    # Add model response
    model_msg = {
        "role": "model",
        "content": answer,
        "timestamp": time.time()
    }
    history.append(model_msg)

    duration = round(time.time() - start, 3)
    logger.info(f"Chat completed | session={session_id} | time={duration}s")

    return ChatResponse(
        session_id=session_id,
        messages=[
            Message(**msg, timestamp=msg["timestamp"])
            for msg in history
        ],
        answer=answer
    )

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    history = sessions[session_id]
    now = time.time()

    return HistoryResponse(
        session_id=session_id,
        messages=[Message(**m, timestamp=m["timestamp"]) for m in history],
        created_at=history[0]["timestamp"],
        last_updated=history[-1]["timestamp"] if history else now
    )

@app.delete("/history/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session deleted → {session_id}")
        return {"message": "Session deleted successfully"}
    raise HTTPException(404, "Session not found")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "sessions_active": len(sessions)
    }

# -------------------------------
# START
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)