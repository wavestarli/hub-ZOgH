"""
web_app.py — 天气 Function Call Web 前端服务

会话模型：
  · 每打开一个页面 → POST /api/session 创建独立会话 + 独立短期记忆
  · 对话走 POST /api/chat
  · 关闭页面 → sendBeacon / DELETE 销毁会话与记忆
  · 心跳超时（默认 45s）兜底清理异常断开的会话
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mode_function_call.agent_core import ShortTermMemory, build_client, run_query

STATIC_DIR = Path(__file__).resolve().parent / "static"
MEMORY_TURNS = 6
HEARTBEAT_TTL_SEC = 45
SWEEP_INTERVAL_SEC = 10


class ChatSession:
    __slots__ = ("session_id", "display_name", "memory", "created_at", "last_seen")

    def __init__(self, session_id: str, display_name: str, max_turns: int):
        self.session_id = session_id
        self.display_name = display_name
        # 每个会话独享一份记忆，关闭即整体丢弃
        self.memory = ShortTermMemory(max_turns=max_turns)
        self.created_at = time.time()
        self.last_seen = self.created_at

    def touch(self) -> None:
        self.last_seen = time.time()

    def to_dict(self) -> dict[str, Any]:
        key = self.session_id
        return {
            "session_id": self.session_id,
            "display_name": self.display_name,
            "memory_turns": self.memory.turn_count(key),
            "memory_max_turns": self.memory.max_turns,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
        }


class SessionManager:
    def __init__(self, max_turns: int = MEMORY_TURNS, ttl: float = HEARTBEAT_TTL_SEC):
        self.max_turns = max_turns
        self.ttl = ttl
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, display_name: str | None = None) -> ChatSession:
        async with self._lock:
            sid = uuid.uuid4().hex[:12]
            name = (display_name or "").strip() or f"访客-{sid[:4]}"
            session = ChatSession(sid, name, self.max_turns)
            self._sessions[sid] = session
            print(f"[session+] {sid} ({name}) | active={len(self._sessions)}", flush=True)
            return session

    async def get(self, session_id: str) -> ChatSession:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.touch()
            return session

    async def destroy(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            # 销毁短期记忆
            session.memory.clear()
            print(
                f"[session-] {session_id} ({session.display_name}) | active={len(self._sessions)}",
                flush=True,
            )
            return True

    async def list_sessions(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    async def sweep_stale(self) -> int:
        now = time.time()
        async with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s.last_seen > self.ttl
            ]
            for sid in stale:
                session = self._sessions.pop(sid)
                session.memory.clear()
                print(
                    f"[session~] timeout {sid} ({session.display_name}) | active={len(self._sessions)}",
                    flush=True,
                )
            return len(stale)


class CreateSessionBody(BaseModel):
    display_name: str | None = Field(default=None, max_length=32)


class ChatBody(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=2000)


manager = SessionManager()
llm_client = None
llm_model = "deepseek-chat"


async def _sweeper():
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SEC)
        try:
            await manager.sweep_stale()
        except Exception as e:
            print(f"[sweeper] error: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, llm_model
    try:
        llm_client, llm_model = build_client("deepseek")
        print(f"[web] LLM ready: {llm_model}", flush=True)
    except RuntimeError as e:
        print(f"[web] WARNING: {e} — /api/chat 将返回 503", flush=True)
        llm_client, llm_model = None, "unavailable"

    task = asyncio.create_task(_sweeper())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Weather Function Call Web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session")
async def create_session(body: CreateSessionBody | None = None):
    name = body.display_name if body else None
    session = await manager.create(name)
    return session.to_dict()


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    try:
        session = await manager.get(session_id)
    except KeyError:
        raise HTTPException(404, "会话不存在或已关闭")
    data = session.to_dict()
    data["history"] = session.memory.get_history(session_id)
    return data


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    ok = await manager.destroy(session_id)
    return {"ok": ok, "session_id": session_id}


@app.post("/api/session/{session_id}/heartbeat")
async def heartbeat(session_id: str):
    try:
        session = await manager.get(session_id)
    except KeyError:
        raise HTTPException(404, "会话不存在或已关闭")
    return {"ok": True, "session_id": session.session_id, "last_seen": session.last_seen}


@app.get("/api/sessions")
async def list_sessions():
    sessions = await manager.list_sessions()
    return {"count": len(sessions), "sessions": sessions}


@app.post("/api/chat")
async def chat(body: ChatBody):
    if llm_client is None:
        raise HTTPException(503, "LLM 未配置：请设置环境变量 DEEPSEEK_API_KEY")

    question = body.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")

    try:
        session = await manager.get(body.session_id)
    except KeyError:
        raise HTTPException(404, "会话不存在或已关闭，请刷新页面")

    # OpenAI 调用是阻塞的，放到线程池避免卡住事件循环
    result = await asyncio.to_thread(
        run_query,
        llm_client,
        llm_model,
        question,
        user_id=session.session_id,
        memory=session.memory,
        verbose=True,
    )
    return {
        "session_id": session.session_id,
        "display_name": session.display_name,
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "elapsed": result["elapsed"],
        "memory_turns": result["memory_turns"],
        "memory_max_turns": result["memory_max_turns"],
    }


@app.post("/api/session/{session_id}/close")
async def close_session_beacon(session_id: str, request: Request):
    """供 navigator.sendBeacon 调用（部分浏览器对 DELETE + beacon 支持不佳）。"""
    # 消耗 body，避免连接半开
    try:
        await request.body()
    except Exception:
        pass
    ok = await manager.destroy(session_id)
    return {"ok": ok}


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="天气 Function Call Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"[web] http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(
        "mode_function_call.web_app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
