# test_llm.py
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

async def main():
    print("Initializing ChatOllama...")
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=2048
    )

    print("\n--- Test 1: Basic Chat ---")
    try:
        resp1 = await llm.ainvoke([HumanMessage(content="What is 2+2?")])
        print(f"Result: {resp1.content!r}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n--- Test 2: System Prompt JSON ---")
    sys_prompt = (
        "You are an intent classification engine. "
        "Analyze the user's message and return a JSON object with keys: "
        "'intent' (SCHEDULE, CHAT, or UNKNOWN), 'parameters' (dict), and 'reasoning'."
    )
    try:
        resp2 = await llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content="Schedule a meeting tomorrow at 5 PM.")
        ])
        print(f"Result: {resp2.content!r}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
