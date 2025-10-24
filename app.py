"""
ACLS Assistant – Single-file FastAPI PoC
RAG + Tool Calling + Simple ACLS Flow Engine + Timers (in-memory)

⚠️ Safety: For development/education only. Do NOT use to replace clinical judgment.
    Update all logic against the latest AHA ACLS guidance and your institution's SOP before any real-world use.

Quickstart
----------
1) Python 3.10+
2) pip install -U fastapi uvicorn pydantic openai chromadb sentence-transformers pypdf
   # Optional if you plan to run a local model server:
   # vLLM (server) or ollama. For vLLM: `pip install vllm` (server run separately).
3) Put your reference docs under ./docs (PDF/TXT/MD)
4) Start the API:
   export LLM_BASE_URL="http://localhost:8000/v1"   # e.g., vLLM or a compatible server
   export LLM_API_KEY="sk-not-used"                  # any string if server ignores it
   export LLM_MODEL="Qwen2.5-72B-Instruct"          # or your loaded model id
   uvicorn app:app --reload --port 9000
5) Ingest your docs:
   curl -X POST http://127.0.0.1:9000/ingest
6) Ask a question / run a scenario:
   curl -X POST http://127.0.0.1:9000/chat \
     -H 'Content-Type: application/json' \
     -d '{"patient_id":"p001","query":"患者無脈VT，剛除顫一次，請幫我開始兩分鐘CPR並計時，並記錄事件"}'

Notes
-----
- This PoC treats the LLM as a language interface only; decisions and timers are enforced by code.
- The flow engine below is a minimal skeleton (shockable vs. non-shockable). Replace with your full FSM/rules.
- RAG returns top chunks; the prompt demands citations; you must ensure your docs include version/source metadata.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- LLM client (OpenAI-compatible; works with vLLM, LM Studio, Ollama OpenAI API, etc.)
from openai import OpenAI

# --- RAG
import chromadb
from chromadb.utils import embedding_functions
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # PDF support optional

# ===========================
# Config
# ===========================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-not-used")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen2.5-72B-Instruct")
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "acls_docs")

# ===========================
# FastAPI app
# ===========================
app = FastAPI(title="ACLS Assistant PoC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# LLM client
# ===========================
llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# ===========================
# RAG components
# ===========================
client = chromadb.PersistentClient(path=DB_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embed_fn)

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf"}


def read_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf not installed; install 'pypdf' to parse PDFs")
        reader = PdfReader(str(path))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def chunk_text(t: str, size: int = 1200, overlap: int = 180) -> List[str]:
    chunks = []
    i = 0
    while i < len(t):
        chunks.append(t[i : i + size])
        i += max(1, size - overlap)
    return chunks


def ingest_dir(folder: str = DOCS_DIR) -> Dict[str, Any]:
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    for f in folder_path.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTS:
            continue
        raw = read_file_text(f)
        text = normalize_text(raw)
        for j, ck in enumerate(chunk_text(text)):
            ids.append(f"{f.name}-{j}")
            docs.append(ck)
            metas.append({
                "source": str(f),
                "chunk": j,
                # Optional: version, date, SOP id, etc.
            })
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return {"chunks_ingested": len(docs), "files": len(set(m["source"] for m in metas))}


def retrieve(query: str, k: int = 4) -> List[Tuple[str, dict]]:
    if not query.strip():
        return []
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

# ===========================
# Simple in-memory store (patients, timers, logs)
# ===========================
@dataclass
class TimerItem:
    label: str
    seconds: int
    start_time: float
    end_time: float
    task: Optional[asyncio.Task] = None
    active: bool = True


@dataclass
class PatientState:
    patient_id: str
    rhythm: Optional[str] = None  # e.g., "VF", "pVT", "PEA", "Asystole", "Unknown"
    shocks_given: int = 0
    last_shock_ts: Optional[float] = None
    last_epi_ts: Optional[float] = None
    last_antiarr_ts: Optional[float] = None
    cpr_active: bool = False
    events: List[dict] = field(default_factory=list)
    timers: Dict[str, TimerItem] = field(default_factory=dict)


PATIENTS: Dict[str, PatientState] = {}


def now_ts() -> float:
    return time.time()


def record_event(patient_id: str, etype: str, detail: str, ts: Optional[str] = None) -> dict:
    st = PATIENTS.setdefault(patient_id, PatientState(patient_id=patient_id))
    iso = ts or datetime.utcnow().isoformat()
    ev = {"time": iso, "type": etype, "detail": detail}
    st.events.append(ev)
    # Lightly update state if notable
    if etype.lower() == "shock":
        st.shocks_given += 1
        st.last_shock_ts = now_ts()
    if etype.lower() == "cpr-start":
        st.cpr_active = True
    if etype.lower() == "cpr-stop":
        st.cpr_active = False
    return ev


async def _timer_worker(patient_id: str, label: str, seconds: int):
    await asyncio.sleep(seconds)
    st = PATIENTS.get(patient_id)
    if not st:
        return
    t = st.timers.get(label)
    if not t:
        return
    t.active = False
    record_event(patient_id, "timer-done", f"{label} completed in {seconds}s")


def start_timer(patient_id: str, label: str, seconds: int) -> dict:
    st = PATIENTS.setdefault(patient_id, PatientState(patient_id=patient_id))
    # cancel existing
    if label in st.timers:
        old = st.timers[label]
        if old.task and not old.task.done():
            old.task.cancel()
    item = TimerItem(label=label, seconds=seconds, start_time=now_ts(), end_time=now_ts() + seconds)
    task = asyncio.create_task(_timer_worker(patient_id, label, seconds))
    item.task = task
    st.timers[label] = item
    record_event(patient_id, "timer-start", f"{label} {seconds}s")
    return {"status": "started", "label": label, "seconds": seconds}


def stop_timer(patient_id: str, label: str) -> dict:
    st = PATIENTS.setdefault(patient_id, PatientState(patient_id=patient_id))
    it = st.timers.get(label)
    if not it:
        return {"status": "not_found", "label": label}
    if it.task and not it.task.done():
        it.task.cancel()
    it.active = False
    record_event(patient_id, "timer-stop", f"{label}")
    return {"status": "stopped", "label": label}


# ===========================
# Minimal ACLS flow engine (skeleton)
# ===========================
class AclsEngine:
    """A *skeleton* finite-step helper. Replace with your validated rules.
    It outputs structured, non-prescriptive actions the UI/LLM can voice.
    """

    CPR_CYCLE_SEC = 120  # 2 minutes
    EPI_WINDOW_SEC = (180, 300)  # 3–5 minutes (range – engine reminds window)

    def propose_next(self, st: PatientState, context: dict) -> dict:
        """Return next actionable steps based on simplified logic.
        This avoids dosing/energy numbers; wire those via SOP config.
        """
        rhythm = (context.get("current_rhythm") or st.rhythm or "Unknown").upper()
        now = now_ts()

        actions: List[dict] = []
        notes: List[str] = []
        citations = [
            {"id": "AHA-ACLS-2025", "section": "Adult Cardiac Arrest – Shockable/Non-shockable algorithms"}
        ]

        # Always prompt high-quality CPR unless actively defibrillating
        if not st.cpr_active:
            actions.append({"action": "start_cpr", "seconds": self.CPR_CYCLE_SEC, "label": "CPR-2min"})
            notes.append("Begin/continue high-quality CPR while preparing next step.")

        if rhythm in {"VF", "PVT", "P-VT", "PULSELESS VT"}:
            # Shockable pathway (generic)
            if st.last_shock_ts is None or (now - st.last_shock_ts) >= self.CPR_CYCLE_SEC:
                actions.append({"action": "prep_defibrillation"})
                actions.append({"action": "deliver_shock", "safety_check": True})
                notes.append("Shockable rhythm pathway: defibrillate when ready; resume CPR immediately after.")
            else:
                notes.append("Within current CPR cycle; continue CPR until rhythm check.")
        else:
            # Non-shockable or unknown: focus on CPR + rhythm check
            actions.append({"action": "rhythm_check_ready"})
            notes.append("Non-shockable/unknown: prioritize CPR; prepare for rhythm check.")

        # Epinephrine reminder window (non-prescriptive)
        if st.last_epi_ts is None:
            actions.append({"action": "epi_window", "seconds": self.EPI_WINDOW_SEC, "label": "Epi-Window"})
        else:
            elapsed = now - st.last_epi_ts
            if elapsed >= self.EPI_WINDOW_SEC[0]:
                actions.append({"action": "epi_window", "seconds": self.EPI_WINDOW_SEC, "label": "Epi-Window"})

        return {
            "patient_id": st.patient_id,
            "rhythm": rhythm,
            "suggested_actions": actions,
            "notes": notes,
            "citations": citations,
        }


ENGINE = AclsEngine()

# ===========================
# Tool schemas (for LLM function-calling)
# ===========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_event",
            "description": "Log a resuscitation event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "etype": {"type":"string","enum":["CPR-Start","CPR-Stop","Shock","RhythmCheck","Drug","Airway","Timer-Start","Timer-Stop","Timer-Done","Note"]},
                    "detail": {"type": "string"},
                    "timestamp": {"type": "string", "description": "ISO-8601", "nullable": True},
                },
                "required": ["patient_id", "etype", "detail"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_timer",
            "description": "Start or reset a named countdown (seconds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "label": {"type": "string"},
                    "seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["patient_id", "label", "seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_timer",
            "description": "Stop a running timer by label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["patient_id", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_next_step",
            "description": "Call the ACLS flow engine to propose next steps (non-prescriptive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "context": {
                        "type": "object",
                        "description": "Optional, current status fields: rhythm, airway, times, etc.",
                    },
                },
                "required": ["patient_id"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是 ACLS 助理。你必須：\n"
    "1) 僅根據引用資料與院內SOP提供資訊；對臨床下一步，請以工具呼叫 (propose_next_step) 取得程式決策。\n"
    "2) 任何紀錄與計時，務必透過工具呼叫 (record_event/start_timer/stop_timer)。\n"
    "3) 回答需條列、簡短，並標示引用來源ID。資訊不足時要明確說明。\n"
    "4) 計時器秒數由系統決定，請在 start_timer 僅提供 label；若提供 seconds 參數，系統可能會覆蓋為安全值。\n"
)

# ===========================
# API models
# ===========================
class ChatRequest(BaseModel):
    patient_id: str
    query: str
    context: Optional[dict] = None
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[dict] = Field(default_factory=list)
    retrieved: List[dict] = Field(default_factory=list)
    events: List[dict] = Field(default_factory=list)
    timers: Dict[str, dict] = Field(default_factory=dict)


# ===========================
# Orchestration helpers
# ===========================

def make_context_block(chunks: List[Tuple[str, dict]]) -> str:
    lines = []
    for i, (doc, meta) in enumerate(chunks, 1):
        src = meta.get("source", "unknown")
        ck = meta.get("chunk", "?")
        lines.append(f"[{i}] {doc}\n(來源: {src}#chunk{ck})")
    return "\n\n".join(lines)


def list_state_snapshot(patient_id: str) -> dict:
    st = PATIENTS.setdefault(patient_id, PatientState(patient_id=patient_id))
    timers = {
        k: {"label": v.label, "seconds": v.seconds, "active": v.active, "eta": max(0, int(v.end_time - now_ts()))}
        for k, v in st.timers.items()
    }
    return {
        "patient_id": patient_id,
        "rhythm": st.rhythm,
        "shocks_given": st.shocks_given,
        "cpr_active": st.cpr_active,
        "timers": timers,
    }


async def run_tools_and_answer(req: ChatRequest) -> ChatResponse:
    # 1) Retrieval
    chunks = retrieve(req.query, k=req.top_k)
    ctx_block = make_context_block(chunks) if chunks else ""

    # 2) Build messages
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {
        "role": "user",
        "content": (
            "以下是可引用的參考資料（僅供佐證）：\n" + ctx_block + "\n\n" +
            "病患狀態摘要：\n" + json.dumps(list_state_snapshot(req.patient_id), ensure_ascii=False) + "\n\n" +
            f"問題/指令：{req.query}\n"
            "請先檢索並引用資料，再透過工具呼叫處理紀錄與計時，以及以 propose_next_step 取得下一步。\n"
            "輸出精簡條列，段尾以 [1][2]… 標示引用。"
        ),
    }

    messages = [sys_msg, user_msg]
    tool_calls_log: List[dict] = []

    # 3) First LLM call
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    # 4) Handle tool calls (single round; can be looped for multi-step)
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            # Execute
            if name == "record_event":
                out = record_event(args["patient_id"], args["etype"], args.get("detail", ""), args.get("timestamp"))
            elif name == "start_timer":
                label = args["label"]
                seconds = int(args.get("seconds", 0) or 0)
                # Server-side guard: enforce safe durations
                if "cpr" in label.lower():
                    seconds = ENGINE.CPR_CYCLE_SEC  # force 2-min CPR cycle
                elif seconds < 10:
                    seconds = 10  # minimum sane timer to avoid accidental 1-2s
                out = start_timer(args["patient_id"], label, seconds)
            elif name == "stop_timer":
                out = stop_timer(args["patient_id"], args["label"])
            elif name == "propose_next_step":
                st = PATIENTS.setdefault(args["patient_id"], PatientState(patient_id=args["patient_id"]))
                out = ENGINE.propose_next(st, args.get("context") or {})
            else:
                out = {"error": f"unknown tool {name}"}

            tool_calls_log.append({"name": name, "args": args, "result": out})

            # Add tool result to messages and continue a second round
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(out, ensure_ascii=False),
            })

        # 5) Second LLM call to produce the final answer
        resp2 = llm.chat.completions.create(model=LLM_MODEL, messages=messages)
        final_text = resp2.choices[0].message.content or ""
    else:
        final_text = msg.content or ""

    st = PATIENTS.setdefault(req.patient_id, PatientState(patient_id=req.patient_id))

    return ChatResponse(
        answer=final_text,
        tool_calls=tool_calls_log,
        retrieved=[{"doc": d, "meta": m} for d, m in chunks],
        events=st.events,
        timers={k: {"label": v.label, "seconds": v.seconds, "active": v.active} for k, v in st.timers.items()},
    )


# ===========================
# FastAPI endpoints
# ===========================
@app.post("/ingest")
def api_ingest():
    return ingest_dir(DOCS_DIR)


@app.get("/patients/{patient_id}/state")
def api_state(patient_id: str):
    return list_state_snapshot(patient_id)


class TimerReq(BaseModel):
    label: str
    seconds: int


@app.post("/patients/{patient_id}/timers")
async def api_start_timer(patient_id: str, body: TimerReq):
    return start_timer(patient_id, body.label, body.seconds)


@app.delete("/patients/{patient_id}/timers/{label}")
async def api_stop_timer(patient_id: str, label: str):
    return stop_timer(patient_id, label)


@app.post("/patients/{patient_id}/events")
async def api_record_event(patient_id: str, body: dict):
    etype = body.get("etype", "note")
    detail = body.get("detail", "")
    ts = body.get("timestamp")
    return record_event(patient_id, etype, detail, ts)


@app.post("/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    return await run_tools_and_answer(req)


# ===========================
# Dev demo curl (commented)
# ===========================
# curl -X POST http://127.0.0.1:9000/ingest
# curl -X POST http://127.0.0.1:9000/chat -H 'Content-Type: application/json' -d '{
#   "patient_id":"p001",
#   "query":"患者無脈VT，剛除顫一次，請幫我開始兩分鐘CPR並計時，並記錄事件",
#   "context":{"current_rhythm":"VF"}
# }'

# End of file