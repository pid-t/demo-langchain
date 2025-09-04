import math
import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def create_vector_store_in_batches(documents, embedding_model, batch_size=100, delay_seconds=1):
    # --- 2. 定义免费套餐的常量 ---
    # Google AI Studio 免费套餐的速率限制
    FREE_TIER_RPM = 60
    # embedding-001 模型的最大批处理大小
    MAX_BATCH_SIZE = 10
    # 计算两次请求之间的安全等待时间（秒）
    # (60秒 / 60次请求) = 1秒/次。增加0.1秒作为网络延迟等的缓冲。
    SECONDS_PER_REQUEST = 5

    print(
        f"速率限制策略：每分钟 {FREE_TIER_RPM} 次请求，批处理大小 {MAX_BATCH_SIZE}，每次请求后等待 {SECONDS_PER_REQUEST:.2f} 秒。"
    )

    vector_store = None
    total_docs = len(documents)
    total_batches = math.ceil(total_docs / MAX_BATCH_SIZE)
    print(f"\n文档总数: {total_docs}，将分为 {total_batches} 个批次处理。")
    for i in range(0, total_docs, MAX_BATCH_SIZE):
        batch = documents[i : i + MAX_BATCH_SIZE]
        current_batch_num = (i // MAX_BATCH_SIZE) + 1

        print(f"--> 正在处理批次 {current_batch_num}/{total_batches} (文档 {i + 1} 到 {i + len(batch)})...")

        if vector_store is None:
            # 对于第一个批次，使用 from_documents 创建向量存储
            # 这是首次 API 调用
            vector_store = InMemoryVectorStore.from_documents(documents=batch, embedding=embedding_model)
        else:
            # 对于后续批次，使用 add_documents 添加到现有存储
            # 这是后续的 API 调用
            vector_store.add_documents(documents=batch)

        print(f"    批次 {current_batch_num} 处理完成。")

        if current_batch_num < total_batches:
            # 如果不是最后一个批次，则等待以遵守速率限制
            print(f"    等待 {SECONDS_PER_REQUEST:.2f} 秒...")
            time.sleep(SECONDS_PER_REQUEST)

    print("\n所有批次处理完毕，向量数据库创建成功！")
    return vector_store


def main2():
    print("start to test PyPDFLoader.")
    file_path = "./nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    print(len(all_splits))
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCpI7qAx60fMpZolNMWiLrKwQyFLKSa6KA"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = create_vector_store_in_batches(all_splits, embeddings, batch_size=5, delay_seconds=10)
    results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
    print(results[0])


if __name__ == "__main__":
    # main()
    main2()
