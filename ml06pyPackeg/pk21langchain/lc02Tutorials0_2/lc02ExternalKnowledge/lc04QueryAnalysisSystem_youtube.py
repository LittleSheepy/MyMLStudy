# Build a Query Analysis System 建立一个查询分析系统
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
"""
    加载文档
"""
print("-"*40,"\n","         加载文档\n","-"*40,"\n")
from langchain_community.document_loaders import YoutubeLoader

urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",      # OpenGPTs
    # "https://www.youtube.com/watch?v=dA1cHGACXCo",      # Building a web RAG chatbot: using LangChain, Exa (prev. Metaphor), LangSmith, and Hosted Langserve
    # "https://www.youtube.com/watch?v=ZcEMLz27sL4",      # Streaming Events: Introducing a new `stream_events` method
    # "https://www.youtube.com/watch?v=hvAPnpSfSGo",      # LangGraph: Multi-Agent Workflows
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",      # Build and Deploy a RAG app with Pinecone Serverless
    # "https://www.youtube.com/watch?v=mmBo8nlu2j0",      # Auto-Prompt Builder (with Hosted LangServe)
    # "https://www.youtube.com/watch?v=rQdibOsL1ps",      # Build a Full Stack RAG App With TypeScript
    # "https://www.youtube.com/watch?v=28lC4fqukoc",      # Getting Started with Multi-Modal LLMs
    # "https://www.youtube.com/watch?v=es-9MgxB-uc",      # SQL Research Assistant
    # "https://www.youtube.com/watch?v=wLRHwKuKvOE",      # Skeleton-of-Thought: Building a New Template from Scratch
    # "https://www.youtube.com/watch?v=ObIltMaRJvY",      # Benchmarking RAG over LangChain Docs
    # "https://www.youtube.com/watch?v=DjuXACWYkkU",      # Building a Research Assistant from Scratch
    # "https://www.youtube.com/watch?v=o7C9ld6Ln-M",      # LangServe and LangChain Templates Webinar
]

docs = []
for url in urls:
    print(url)
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
print([doc.metadata["title"] for doc in docs])
print(docs[0].metadata)
print(docs[0].page_content[:500])


"""
    索引文档
"""
print("-"*40,"\n","         索引文档\n","-"*40,"\n")
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from transformers.utils import is_torch_cuda_available, is_torch_mps_available
# EMBEDDING_DEVICE = "cuda" if is_torch_cuda_available() else "mps" if is_torch_mps_available() else "cpu"
# embeddings_model = HuggingFaceEmbeddings(model_name='D:\models\m3e-base', model_kwargs={'device': EMBEDDING_DEVICE})
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)
embeddings = DashScopeEmbeddings()
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)

"""********************
    不使用查询分析的检索
********************"""
print("-"*40,"\n","         不使用查询分析的检索\n","-"*40,"\n")
search_results = vectorstore.similarity_search("how do I build a RAG agent")
print(search_results[0].metadata["title"])      # Build and Deploy a RAG app with Pinecone Serverless
print(search_results[0].page_content[:500])

search_results = vectorstore.similarity_search("videos on RAG published in 2023")
print(search_results[0].metadata["title"])
print(search_results[0].metadata["publish_date"])
print(search_results[0].page_content[:500])

"""********************
    查询分析
********************"""
print("-"*40,"\n","         查询分析\n","-"*40,"\n")
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class Search(BaseModel):
    """Search over a database of tutorial videos about a software library.
    搜索关于软件库的教程视频数据库"""
    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# 使用 with_structured_output 方法来获取结构化的输出
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

query_analyzer.invoke("how do I build a RAG agent")
# -> Search(query='build RAG agent', publish_year=None)

query_analyzer.invoke("videos on RAG published in 2023")
# -> Search(query='RAG', publish_year=2023)

"""
    使用查询分析的检索
"""
print("-"*40,"\n","         使用查询分析的检索\n","-"*40,"\n")
from typing import List

from langchain_core.documents import Document
def retrieval(search: Search) -> List[Document]:
    if search.publish_year is not None:
        # This is syntax specific to Chroma,
        # the vector database we are using.
        _filter = {"publish_year": {"$eq": search.publish_year}}
    else:
        _filter = None
    return vectorstore.similarity_search(search.query, filter=_filter)
retrieval_chain = query_analyzer | retrieval

results = retrieval_chain.invoke("RAG tutorial published in 2023")

print([(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results])








pass