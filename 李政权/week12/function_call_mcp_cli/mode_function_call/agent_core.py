"""
agent_core.py — Function Call 天气助手核心（CLI / Web 共用）

- ShortTermMemory：按会话/用户隔离的短期记忆
- run_query：单轮 tool_call 闭环
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weather_backend import get_weather

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    }
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_weather": get_weather,
}

SYSTEM_PROMPT = (
    "你是一名天气分析助手。回答用户涉及天气的问题时，请调用 get_weather。"
    "你可以一次调用多个工具。请结合对话历史理解指代（如「那上海呢」「刚才那个城市」）。"
)

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def build_client(provider: str = "deepseek"):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        raise RuntimeError(f"未设置 {provider.upper()}_API_KEY 环境变量")
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


class ShortTermMemory:
    """按 session/user 隔离的短期对话记忆（滑动窗口，线程安全）。"""

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.RLock()

    def list_users(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions.keys())

    def get_history(self, user_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(user_id, []))

    def turn_count(self, user_id: str) -> int:
        return len(self.get_history(user_id)) // 2

    def append_turn(self, user_id: str, question: str, answer: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(user_id, [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            max_msgs = self.max_turns * 2
            if len(history) > max_msgs:
                self._sessions[user_id] = history[-max_msgs:]

    def clear(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(user_id, None)

    def summary(self, user_id: str) -> str:
        turns = self.turn_count(user_id)
        return f"用户={user_id} | 已记忆 {turns}/{self.max_turns} 轮"


def run_query(
    client,
    model: str,
    question: str,
    *,
    user_id: str = "default",
    memory: ShortTermMemory | None = None,
    verbose: bool = True,
    max_tool_rounds: int = 3,
) -> dict[str, Any]:
    """单次查询：注入短期记忆 → tool_call → 执行 → 回填 → 回答 → 写入记忆。"""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory is not None:
        messages.extend(memory.get_history(user_id))
    messages.append({"role": "user", "content": question})

    t0 = time.time()
    tool_call_log: list[dict[str, Any]] = []
    prefix = f"[{user_id}] "
    answer = ""

    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                _safe_print(f"  {prefix}→ [tool] {name}({args})")

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                _safe_print(f"  {prefix}  ↩ {preview}{'...' if len(result or '') > 120 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        resp = client.chat.completions.create(model=model, messages=messages)
        answer = resp.choices[0].message.content or ""

    elapsed = time.time() - t0
    if verbose:
        _safe_print(f"  {prefix}→ [llm] 最终回答（{elapsed:.1f}s）")

    if memory is not None and answer:
        memory.append_turn(user_id, question, answer)

    return {
        "user_id": user_id,
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "memory": memory.summary(user_id) if memory else None,
        "memory_turns": memory.turn_count(user_id) if memory else 0,
        "memory_max_turns": memory.max_turns if memory else 0,
    }
