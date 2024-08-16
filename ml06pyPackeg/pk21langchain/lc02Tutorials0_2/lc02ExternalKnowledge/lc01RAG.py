# Retrieval Augmented Generation 检索增强生成
import os

import bs4
from langchain import hub
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

llm = ChatTongyi(model="qwen-max")

"""********************
        索引
********************"""
print("-"*40,"\n","         索引\n","-"*40,"\n")
# 1.索引：加载
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

loader = TextLoader(r'txt_cn.txt', encoding='utf8')
docs = loader.load()
# def read_txt_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     return content
# file_content = read_txt_file('txt.txt')
# docs = [Document(page_content=file_content)]
print(len(docs[0].page_content))    # 43131
print(docs[0].page_content[:50])

# 2.索引：拆分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))                      # 66
print(len(all_splits[0].page_content))      # 969
print(all_splits[10].metadata)              # {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 7056}

# 3.索引：储存
vectorstore = Chroma.from_documents(documents=all_splits, embedding=DashScopeEmbeddings())

"""********************
        检索和生成
********************"""
print("-"*40,"\n","         检索和生成\n","-"*40,"\n")
# 1.检索
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
print(len(retrieved_docs))
print(retrieved_docs[0].page_content)

# 2.生成
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
print(example_messages)
print(example_messages[0].content)
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: filler question 
Context: filler context 
Answer:
"""

# 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)

result = rag_chain.invoke("What is Task Decomposition?")
print(result)

# 其他
promptEx = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)
result = promptEx.invoke("What is Task Decomposition?")
print(result.messages[0].content)






pass

