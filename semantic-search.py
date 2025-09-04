import math
import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def create_vector_store_in_batches(documents, embedding_model, batch_size=100, delay_seconds=1):
    print(
        f"速率限制策略：每分钟 {math.ceil(60 / delay_seconds)} 次请求，批处理大小 {batch_size}，每次请求后等待 {delay_seconds:.2f} 秒。"
    )

    vector_store = None
    total_docs = len(documents)
    total_batches = math.ceil(total_docs / batch_size)
    print(f"\n文档总数: {total_docs}，将分为 {total_batches} 个批次处理。")
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"\n处理第 {i // batch_size + 1} 批次，包含 {len(batch)} 个文档...")
        current_batch_num = (i // batch_size) + 1

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
            print(f"    等待 {delay_seconds:.2f} 秒...")
            time.sleep(delay_seconds)

    print("\n所有批次处理完毕，向量数据库创建成功！")
    return vector_store


def main():
    print("start to test PyPDFLoader.")
    file_path = "./nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # print(len(docs))
    # print(f"{docs[0].page_content[:200]}\n")
    # print(docs[0].metadata)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    print(len(all_splits))
    embeddings = OpenAIEmbeddings(
        openai_api_base="https://api.siliconflow.cn/v1",  # SiliconFlow 的 endpoint
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 从环境变量读更安全
        model="BAAI/bge-m3",  # 任意 SiliconFlow 支持的模型
        # dimensions=None,  # 如需要固定维度，可传 int
    )
    vector_store = create_vector_store_in_batches(all_splits, embeddings, batch_size=10, delay_seconds=1)
    results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
    print(results[0])


if __name__ == "__main__":
    main()
