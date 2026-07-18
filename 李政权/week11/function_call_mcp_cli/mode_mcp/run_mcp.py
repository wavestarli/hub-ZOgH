"""
run_mcp.py — 方式二：MCP Host（连接多 Server，单轮闭环调用，可循环调用版）

改进：
  1. 支持持续接收用户输入，无限次调用
  2. 保持原有的 MCP 机制不变
  3. 添加友好的交互提示和退出功能
"""

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

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


# ── Server 配置 ────────────────────────────────────────────────────────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


# ── 连接所有 Server ────────────────────────────────────────────────────────

async def connect_all_servers(stack: AsyncExitStack):
    """
    连接所有 MCP Server，返回 (tool_registry, openai_tools)：
      tool_registry : tool_name → (ClientSession, server_label)
      openai_tools  : 转成 OpenAI tools schema 的列表
    """
    print("🔄 正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)

    print(f"\n✅ 共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


# ── 系统提示词 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气分析助手。回答用户涉及天气时，请调用 get_weather，本回合你可以一次调用多个工具。"
)


async def run_single_query(client, model: str, question: str,
                           tool_registry: dict, openai_tools: list[dict],
                           verbose: bool = True) -> dict:
    """单轮查询闭环：提问 → 模型输出 tool_call → 路由到 Server 执行 → 回填 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # 第一次请求：带上 tools，让模型决定是否调用工具
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=openai_tools, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [mcp] {name}({args})")

            # 查路由表找到对应 Server 的 ClientSession
            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # 第二次请求：模型看到工具结果，生成最终回答
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


async def interactive_mode(client, model: str, tool_registry: dict, openai_tools: list[dict]):
    """
    交互式循环模式：持续接收用户输入，无限次调用天气查询。
    """
    print("=" * 70)
    print("🌤️  AI 天气助手（MCP 模式 · 可循环调用）")
    print("=" * 70)
    print(f"📡 Provider: deepseek | Model: {model}")
    print(f"🔧 可用工具：{', '.join(t['function']['name'] for t in openai_tools)}")
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

            result = await run_single_query(client, model, user_input,
                                           tool_registry, openai_tools)

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


async def demo_mode(client, model: str, tool_registry: dict, openai_tools: list[dict]):
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
        print("-" * 60)

        result = await run_single_query(client, model, question,
                                        tool_registry, openai_tools)

        print("-" * 60)
        print(f"🤖 AI 回答：\n{result['answer']}")
        print(f"⏱️  耗时：{result['elapsed']:.2f}秒\n")
        print("=" * 70 + "\n")


# ── 入口 ───────────────────────────────────────────────────────────────────

async def main_async(provider: str, question: str | None, demo: bool,
                     verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        if demo:
            # 演示模式
            await demo_mode(client, model, tool_registry, openai_tools)
        elif question:
            # 单问题模式（向后兼容）
            if not verbose:
                pass
            result = await run_single_query(client, model, question,
                                            tool_registry, openai_tools,
                                            verbose=verbose)

            if as_json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(f"\n答案：{result['answer']}")
        else:
            # 默认：交互式循环模式
            await interactive_mode(client, model, tool_registry, openai_tools)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP（可循环调用）")
    parser.add_argument("--question", "-q", help="单个问题（非交互模式）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--demo", action="store_true", help="演示模式")
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    asyncio.run(main_async(args.provider, args.question, args.demo,
                          verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()