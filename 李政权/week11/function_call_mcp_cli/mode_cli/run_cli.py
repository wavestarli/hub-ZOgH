"""
run_cli.py — 方式三：CLI（命令行即工具），两种形态，可循环调用版

改进：
  1. 支持持续接收用户输入，无限次调用
  2. 保持原有的 named/bash 两种形态不变
  3. 添加友好的交互提示和退出功能
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

# fincli 真实命令路径
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": "sk-44064086d6d24fb98e5959a062f004e1",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    }
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 形态 A：具名 run_cli ───────────────────────────────────────────────────

NAMED_COMMANDS = {
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}


def run_named(command: str, args: dict) -> str:
    """形态 A：按白名单拼出 argv，子进程执行，返回 stdout。"""
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


# ── 形态 B：通用 run_bash（沙箱）──────────────────────────────────────────

DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bdeltree\b",
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"[>;]\s*(?:rm|del|format)\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
    r"\bsudo\b", r"\bchmod\b.*-R", r"\bchown\b.*-R",
    r"\bnc\b", r"\bnetcat\b",
    r"/etc/passwd", r"/etc/shadow",
    r"\bTaskkill\b", r"\bStop-Process\b",
]

ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}


def sandbox_check(command: str) -> str | None:
    """返回 None 表示通过；返回字符串表示拒绝原因。"""
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    """形态 B：模型生成的 shell 命令，经沙箱检查后在锁定工作目录执行。"""
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


# ── 两种形态各自的 tools schema ───────────────────────────────────────────

NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查天气（weather）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "weather（查天气，需 city）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。weather: {city}",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "可用工具 fincli（一条真实命令）："
                "fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3；"
                "fincli list-companies；"
                "fincli weather --city 宁德。"
                "危险命令（rm/del/format/sudo/curl|sh 等）会被拦截；只允许白名单可执行文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]

# 形态 → (schema, executor)
MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}


# ── 系统提示词 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_NAMED = (
    "你是一名天气分析助手。通过 run_cli 工具调用预批准命令查天气。"
    "本回合可一次调用多个工具。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名天气分析助手。通过 run_bash 工具在沙箱里执行 fincli 命令查与天气。"
    "查天气：fincli weather --city 南京。"
    "回答必须依据命令返回的原文，不要编造。本回合可一次调用多个工具。"
)


def run_single_query(client, model: str, question: str, mode: str,
                     verbose: bool = True) -> dict:
    """单轮查询闭环。"""
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    resp = client.chat.completions.create(
        model=model, messages=messages, tools=tools_schema, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")
            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })

        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


def interactive_mode(client, model: str, mode: str):
    """
    交互式循环模式：持续接收用户输入，无限次调用。
    """
    mode_label = "Named (具名)" if mode == "named" else "Bash (通用)"

    print("=" * 70)
    print(f"🌤️  AI 天气助手（CLI/{mode_label} 模式 · 可循环调用）")
    print("=" * 70)
    print(f"📡 Provider: deepseek | Model: {model}")
    print(f"⚙️  模式：{mode_label}")
    print(f"🔧 底层命令：{FINCLI_LABEL}")
    print("💡 输入城市名查询天气，输入 'q' 退出程序\n")

    query_count = 0

    while True:
        try:
            user_input = input("🧑 请输入问题：").strip()

            # 检查退出条件
            if user_input.lower() in ('q', 'quit', 'exit', '退出'):
                print("👋 感谢使用，再见！")
                break

            # 检查空输入
            if not user_input:
                print("⚠️  问题不能为空，请重新输入\n")
                continue

            query_count += 1
            print(f"\n📝 第 {query_count} 次查询")
            print(f"🔍 正在处理：'{user_input}'")
            print("-" * 60)

            result = run_single_query(client, model, user_input, mode)

            print("-" * 60)
            print(f"🤖 AI 回答：\n{result['answer']}")

            if result['tool_calls']:
                print(f"\n📊 本次调用了 {len(result['tool_calls'])} 个工具")
                for tc in result['tool_calls']:
                    print(f"   • {tc['name']}({tc['args']})")

            print(f"⏱️  耗时：{result['elapsed']:.2f}秒\n")
            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}\n")


def demo_mode(client, model: str, mode: str):
    """
    演示模式：展示两种形态的示例。
    """
    mode_label = "Named" if mode == "named" else "Bash"
    print(f"🎯 演示模式：CLI/{mode_label}\n")

    demo_questions = [
        "宁德今天天气怎么样？",
        "北京和上海的天气分别如何？",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"📝 示例 {i}/{len(demo_questions)}")
        print(f"🧑 问题：'{question}'")
        print("-" * 60)

        result = run_single_query(client, model, question, mode)

        print("-" * 60)
        print(f"🤖 AI 回答：\n{result['answer']}")
        print(f"⏱️  耗时：{result['elapsed']:.2f}秒\n")
        print("=" * 70 + "\n")


# ── 入口 ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式三：CLI（可循环调用）")
    parser.add_argument("--mode", default="named", choices=["named", "bash"],
                       help="运行模式：named（具名）/ bash（通用）")
    parser.add_argument("--question", "-q", help="单个问题（非交互模式）")
    parser.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)

    if args.demo:
        # 演示模式
        demo_mode(client, model, args.mode)
    elif args.question:
        # 单问题模式（向后兼容）
        if not args.quiet:
            print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n")
        result = run_single_query(client, model, args.question, args.mode,
                                  verbose=not args.quiet)

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"\n答案：{result['answer']}")
    else:
        # 默认：交互式循环模式
        interactive_mode(client, model, args.mode)


if __name__ == "__main__":
    main()