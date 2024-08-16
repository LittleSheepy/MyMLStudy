# Vector stores and retrievers 矢量存储和检索器
import asyncio
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
model = Tongyi()

"""
    文件
"""
print("-"*40, "\n    文件\n", "-"*40)
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


"""
    Vector stores
"""
print("-"*40, "\n    Vector stores\n", "-"*40)
# 创建 Chroma 向量存储
vectorstore = Chroma.from_documents(
    documents,
    embedding=DashScopeEmbeddings(),
)
result = vectorstore.similarity_search("cat")
print(result)

# 异步查询
async def asnc_search():
    result = await vectorstore.asimilarity_search("猫")
    print(result)
asyncio.run(asnc_search())

result = vectorstore.similarity_search_with_score("cat")
print(result)

embedding = DashScopeEmbeddings().embed_query("cat")
result = vectorstore.similarity_search_by_vector(embedding)
print(result)

embedding = DashScopeEmbeddings().embed_query("猫")
result = vectorstore.similarity_search_by_vector(embedding)
print(result)

"""
    Retrievers 检索器
"""
print("-"*40, "\n    Retrievers 检索器\n", "-"*40)
# 构建一个 Runnable 类型
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # 选择顶部结果
result = retriever.batch(["猫", "鲨鱼"])
print(result)

retriever = vectorstore.as_retriever(
    search_type="similarity",       # similarity：相似性 mmr：最大边际相关性 similarity_score_threshold
    search_kwargs={"k": 1},
)
result = retriever.batch(["猫", "鲨鱼"])
print(result)

# 最简单的检索增强生成 (RAG)
message = """
仅使用提供的上下文回答此问题。
{question}

上下文:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])
# 构建 RAG 链
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

# 使用 RAG 链并打印结果
response = rag_chain.invoke("告诉我关于猫的事")
print(response)

response = rag_chain.invoke("tell me about cats")
print(response)

message = """
仅使用提供的上下文回答此问题。
告诉我关于猫的事

上下文:
Cats are independent pets that often enjoy their own space.
"""

response = model.invoke(message)
print(response)

pass





























