"""
run_function_call.py — 方式一：Function Call（多人并发循环 + 短期记忆）

能力：
  1. 持续循环接收提问，支持无限次天气查询
  2. 按用户隔离的短期对话记忆（滑动窗口）
  3. 多用户可同时并发循环调用（线程安全）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mode_function_call.agent_core import (  # noqa: E402
    PROVIDERS,
    ShortTermMemory,
    _safe_print,
    build_client,
    run_query,
)


def _parse_user_input(raw: str, current_user: str) -> tuple[str, str]:
    """
    解析输入，支持切换用户：
      @alice 北京天气怎么样？
      alice: 北京天气怎么样？
      北京天气怎么样？          → 沿用 current_user
    """
    text = raw.strip()
    if text.startswith("@") and " " in text:
        user, question = text[1:].split(" ", 1)
        return user.strip() or current_user, question.strip()
    if ":" in text:
        maybe_user, question = text.split(":", 1)
        maybe_user = maybe_user.strip()
        if maybe_user and " " not in maybe_user and len(maybe_user) <= 32:
            return maybe_user, question.strip()
    return current_user, text


def interactive_mode(client, model: str, memory: ShortTermMemory):
    """交互式循环：多用户共享同一进程，每人独立短期记忆。"""
    current_user = "user1"
    print("=" * 60)
    print("AI 天气助手（Function Call · 多人循环 + 短期记忆）")
    print("=" * 60)
    print(f"Provider: deepseek | Model: {model}")
    print(f"短期记忆窗口：最近 {memory.max_turns} 轮 / 每用户")
    print("用法：")
    print("  · 直接提问（沿用当前用户）")
    print("  · @用户名 问题   或   用户名: 问题   （切换用户）")
    print("  · /who  /users  /memory  /clear  /clearall  /user 名称  /q")
    print()

    while True:
        try:
            raw = input(f"[{current_user}] 请输入：").strip()
            if not raw:
                print("问题不能为空，请重新输入\n")
                continue

            cmd = raw.lower()
            if cmd in ("q", "quit", "exit", "/q", "退出"):
                print("感谢使用，再见！")
                break
            if cmd in ("/who",):
                print(f"当前用户：{current_user} | {memory.summary(current_user)}\n")
                continue
            if cmd in ("/users",):
                users = memory.list_users() or [current_user]
                print("已知用户：" + ", ".join(users) + "\n")
                continue
            if cmd in ("/memory",):
                history = memory.get_history(current_user)
                if not history:
                    print(f"[{current_user}] 暂无短期记忆\n")
                else:
                    print(f"── [{current_user}] 短期记忆（{len(history)//2} 轮）──")
                    for i in range(0, len(history), 2):
                        u = history[i].get("content", "")
                        a = history[i + 1].get("content", "") if i + 1 < len(history) else ""
                        print(f"  Q: {u}")
                        print(f"  A: {a[:160]}{'...' if len(a) > 160 else ''}")
                    print()
                continue
            if cmd in ("/clear",):
                memory.clear(current_user)
                print(f"已清空 [{current_user}] 的短期记忆\n")
                continue
            if cmd in ("/clearall",):
                memory.clear()
                print("已清空全部用户的短期记忆\n")
                continue
            if cmd.startswith("/user "):
                name = raw.split(" ", 1)[1].strip()
                if name:
                    current_user = name
                    print(f"已切换到用户：{current_user}\n")
                continue

            user_id, question = _parse_user_input(raw, current_user)
            if not question:
                print("问题不能为空，请重新输入\n")
                continue
            current_user = user_id

            print(f"\n[{user_id}] 正在处理：'{question}'")
            print("-" * 50)
            result = run_query(client, model, question, user_id=user_id, memory=memory)
            print("-" * 50)
            print(f"[{user_id}] AI 回答：\n{result['answer']}")
            if result["tool_calls"]:
                print(f"\n本次调用了 {len(result['tool_calls'])} 个工具")
                for tc in result["tool_calls"]:
                    print(f"   · {tc['name']}({tc['args']})")
            print(f"耗时：{result['elapsed']:.2f}s | {result['memory']}\n")
            print("=" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n程序已退出，再见！")
            break
        except Exception as e:
            print(f"发生错误：{e}\n")


def multi_user_concurrent_mode(
    client,
    model: str,
    memory: ShortTermMemory,
    rounds: int = 2,
):
    """演示：多个用户各自循环提问，线程池并发执行；每人记忆互不干扰。"""
    scripts: dict[str, list[str]] = {
        "alice": [
            "宁德今天天气怎么样？",
            "那北京呢？",
        ],
        "bob": [
            "上海和广州的天气分别如何？",
            "哪个更适合出门？",
        ],
        "carol": [
            "杭州现在下雨吗？",
            "未来三天呢？",
        ],
    }
    for uid, qs in list(scripts.items()):
        scripts[uid] = (qs * ((rounds + len(qs) - 1) // len(qs)))[:rounds]

    print("=" * 60)
    print("多人同时循环调用演示（线程并发 + 短期记忆隔离）")
    print("=" * 60)
    print(f"用户：{', '.join(scripts)} | 每人 {rounds} 轮\n")

    def _user_loop(user_id: str, questions: list[str]) -> list[dict]:
        results = []
        for i, q in enumerate(questions, 1):
            _safe_print(f"\n>>> [{user_id}] 第 {i}/{len(questions)} 轮：{q}")
            r = run_query(client, model, q, user_id=user_id, memory=memory, verbose=True)
            _safe_print(f"<<< [{user_id}] {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")
            _safe_print(f"    ({r['elapsed']:.2f}s | {r['memory']})")
            results.append(r)
            time.sleep(0.05)
        return results

    t0 = time.time()
    all_results: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=len(scripts)) as pool:
        futures = {
            pool.submit(_user_loop, uid, qs): uid
            for uid, qs in scripts.items()
        }
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                all_results[uid] = fut.result()
            except Exception as e:
                _safe_print(f"[{uid}] 失败：{e}")
                all_results[uid] = []

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"全部完成，总墙钟时间 {elapsed:.1f}s（并发）")
    print("各用户记忆状态：")
    for uid in scripts:
        print(f"  · {memory.summary(uid)}")
    print("=" * 60)
    return all_results


def demo_mode(client, model: str, memory: ShortTermMemory):
    """单用户演示：含指代追问，验证短期记忆。"""
    print("演示模式：单用户多轮（验证短期记忆）\n")
    questions = [
        "宁德今天天气怎么样？",
        "北京和上海的天气分别如何？",
        "那刚才宁德和北京比，哪里更冷？",
    ]
    user_id = "demo"
    for i, question in enumerate(questions, 1):
        print(f"示例 {i}/{len(questions)} | {memory.summary(user_id)}")
        print(f"问题：'{question}'")
        print("-" * 50)
        result = run_query(client, model, question, user_id=user_id, memory=memory)
        print("-" * 50)
        print(f"AI 回答：\n{result['answer']}")
        print(f"耗时：{result['elapsed']:.2f}s\n")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="方式一：Function Call（多人并发循环 + 短期记忆）"
    )
    parser.add_argument("--question", "-q", help="单个问题（非交互模式）")
    parser.add_argument("--user", "-u", default="default", help="用户 ID（配合 -q）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--demo", action="store_true", help="单用户多轮演示（含记忆）")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="多人同时循环调用演示（线程并发）",
    )
    parser.add_argument("--rounds", type=int, default=2, help="--multi 时每人提问轮数")
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=6,
        help="每位用户短期记忆保留的对话轮数（默认 6）",
    )
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    try:
        client, model = build_client(args.provider)
    except RuntimeError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    memory = ShortTermMemory(max_turns=args.memory_turns)

    if args.multi:
        multi_user_concurrent_mode(client, model, memory, rounds=args.rounds)
    elif args.demo:
        demo_mode(client, model, memory)
    elif args.question:
        if not args.quiet:
            print(f"[Function Call] provider={args.provider} model={model} user={args.user}\n")
        result = run_query(
            client,
            model,
            args.question,
            user_id=args.user,
            memory=memory,
            verbose=not args.quiet,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"\n答案：{result['answer']}")
    else:
        interactive_mode(client, model, memory)


if __name__ == "__main__":
    main()
