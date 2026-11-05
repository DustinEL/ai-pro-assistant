from operator import itemgetter
from pathlib import Path
import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 在项目根目录 
load_dotenv(".env")
load_dotenv(".env")

llm = init_chat_model(
    model="deepseek-v4.1-flash",
    model_provider="openai",
    api_key=os.getenv("CIYUANSHENDS_API_KEY"),
    base_url="https://api.ciyuanshen.top/v1",
)

prompt_template = """
        请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，
        如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。

        文本内容：
        {context}

        问题：{question}

        回答：
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

embedding_model = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("aliQwen-api"),
)


docx_path =  "alibaba-java.docx"
if not docx_path.exists():
    docx_path =  "alibaba-java.docx"

documents = Docx2txtLoader(str(docx_path)).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    length_function=len,
)
texts = text_splitter.split_documents(documents)

print("文档个数:", len(texts))

REDIS_URL = "redis://localhost:26379"
INDEX_NAME = "my_index3"

# 如果索引是用别的 embedding 维度建的，先删掉再跑：
# redis-cli -p 26379 FT.DROPINDEX my_index3 DD

config = RedisConfig(index_name=INDEX_NAME, redis_url=REDIS_URL)

vec_store = RedisVectorStore.from_documents(
    documents=texts,
    embedding=embedding_model,
    config=config,
)

# retriever: 输入 str -> 输出 list[Document]
retriever = vec_store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    """把 list[Document] 拼成一段纯文本，供 prompt 的 {context} 使用。"""
    return "\n\n".join(doc.page_content for doc in docs)


# 链头是 RunnableParallel：同一个输入分发给 context 和 question 两个分支。
# 这里输入是字符串，所以 retriever 拿到字符串，Passthrough 也把字符串交给 {question}。
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

question = "00000和A0001分别是什么意思"
result = rag_chain.invoke(question)

print("\n问题:", question)
print("\n回答:", result.content)
