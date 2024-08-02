# Retrieval Augmented Generation 检索增强生成
import os

import bs4
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

llm = Tongyi()









































