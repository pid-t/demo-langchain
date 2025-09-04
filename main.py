import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()


def main():
    memory = MemorySaver()
    model = init_chat_model(
        model=os.getenv("MODEL"),
        model_provider="openai",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )

    search = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    tools = [search]

    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    config: RunnableConfig = {"configurable": {"thread_id": "abc123"}}

    print("你好，我是你的天气助手(AI 版).")
    while True:
        user_input = input("👩‍🎤 ：")
        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ["退出", "exit"]:
            print("再见！")
            break

        print("🤖 :", end="")
        input_message = {"role": "user", "content": user_input}
        # streaming output
        for step, metadata in agent_executor.stream({"messages": [input_message]}, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                print(text, end="")
        print()


if __name__ == "__main__":
    main()
