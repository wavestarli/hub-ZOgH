"""
run_function_call.py — 方式一：Function Call（模型原生函数调用，可循环调用版）

改进：
  1. 支持持续接收用户输入，无限次调用
  2. 保持原有的 Function Call 机制不变
  3. 添加友好的交互提示和退出功能
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import
sys.path.insert(0, str(Path(__file__).parent.parent))

# 注意：根据你的实际项目结构调整导入路径
# 如果 run_function_call.py 在 mode_function_call/ 目录下
try:
    from src.weather_backend import get_weather
except ImportError:
    # 如果直接运行本文件，可能需要调整导入方式
    # 这里假设 weather_backend.py 在 src 目录下
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from ..src.weather_backend import get_weather

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        # "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
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


# ── 工具 Schema ────────────────────────────────────────────────────────────

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

# ── 工具调度表 ─────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "get_weather": get_weather,
}

# ── 系统提示词 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气分析助手。回答用户涉及关于天气的问题时，请调用get_weather，本回合你可以一次调用多个工具。"
)


def run_single_query(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    单轮查询闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 最终回答。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # 第一次请求：带上 tools，让模型决定是否调用工具
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 如果模型输出了 tool_calls → 逐个执行后端函数
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")

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
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 第二次请求：模型看到工具结果，生成最终回答
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")

    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


def interactive_mode(client, model: str):
    """
    交互式循环模式：持续接收用户输入，无限次调用天气查询。
    """
    print("=" * 60)
    print("🌤️  AI 天气助手（Function Call 模式 · 可循环调用）")
    print("=" * 60)
    print(f"📡 Provider: deepseek | Model: {model}")
    print("💡 输入城市名查询天气，输入 'q' 退出程序\n")

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

            print(f"\n🔍 正在处理：'{user_input}'")
            print("-" * 50)

            result = run_single_query(client, model, user_input)

            print("-" * 50)
            print(f"🤖 AI 回答：\n{result['answer']}")

            if result['tool_calls']:
                print(f"\n📊 本次调用了 {len(result['tool_calls'])} 个工具")
                for tc in result['tool_calls']:
                    print(f"   • {tc['name']}({tc['args']})")

            print(f"⏱️  耗时：{result['elapsed']:.2f}秒\n")
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}\n")


def demo_mode(client, model: str):
    """
    演示模式：展示并行工具调用的示例。
    """
    print("🎯 演示模式：并行工具调用\n")

    demo_questions = [
        "宁德今天天气怎么样？",
        "北京和上海的天气分别如何？",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"📝 示例 {i}/{len(demo_questions)}")
        print(f"🧑 问题：'{question}'")
        print("-" * 50)

        result = run_single_query(client, model, question)

        print("-" * 50)
        print(f"🤖 AI 回答：\n{result['answer']}")
        print(f"⏱️  耗时：{result['elapsed']:.2f}秒\n")
        print("=" * 60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call（可循环调用）")
    parser.add_argument("--question", "-q", help="单个问题（非交互模式）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)

    # 根据参数选择运行模式
    if args.demo:
        # 演示模式
        demo_mode(client, model)
    elif args.question:
        # 单问题模式（向后兼容）
        if not args.quiet:
            print(f"[Function Call] provider={args.provider} model={model}\n")
        result = run_single_query(client, model, args.question, verbose=not args.quiet)

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"\n答案：{result['answer']}")
    else:
        # 默认：交互式循环模式
        interactive_mode(client, model)


if __name__ == "__main__":
    main()
