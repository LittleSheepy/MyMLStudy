# Retrieval Augmented Generation 检索增强生成
import os

from langchain_community.llms import Tongyi

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

llm = Tongyi()
"""********************
        创建检索器
********************"""
# 1.加载
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(len(docs[0].page_content))    # 43131

# 2.拆分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))                      # 66
print(len(all_splits[0].page_content))      # 969
print(all_splits[10].metadata)              # {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 7056}

# 3.储存
vectorstore = Chroma.from_documents(documents=all_splits, embedding=DashScopeEmbeddings())
"""********************
        检索和生成
********************"""
# 1.检索
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
len(retrieved_docs)
print(retrieved_docs[0].page_content)

# 2.生成
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
print(example_messages)
print(example_messages[0].content)

# 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

# example_messages = prompt.invoke(
#     {"context": "filler context", "question": "filler question"}
# ).to_messages()


for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)















































