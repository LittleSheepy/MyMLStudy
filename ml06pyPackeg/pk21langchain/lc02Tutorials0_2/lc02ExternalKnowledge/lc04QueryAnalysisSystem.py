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
from langchain_community.document_loaders import BiliBiliLoader

SESSDATA = "***************************************"
BUVID3 = "**************************************"
BILI_JCT = "******************************************"

loader = BiliBiliLoader(
    [
        "https://www.bilibili.com/video/BV1PZ421S7VF/?spm_id_from=333.1007.tianma.1-2-2.click",
    ],
    sessdata=SESSDATA,
    bili_jct=BILI_JCT,
    buvid3=BUVID3,
)
docs = loader.load()
print(docs[0].metadata)
import datetime

# Add some additional metadata: what year the video was published
for doc in docs:
    doc.metadata["pub_year"] = int(
        datetime.datetime.strptime(
            datetime.datetime.fromtimestamp(doc.metadata["pubdate"]).strftime('%Y-%m-%d %H:%M:%S'),
            "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )

print([doc.metadata["title"] for doc in docs])
# sub_list = sub["subtitles"]
# if sub_list:
#     sub_url = sub_list[0]["subtitle_url"]
#     if not sub_url.startswith("http"):
#         sub_url = "https:" + sub_url
#     response = requests.get(sub_url)
#
# from langchain_community.document_loaders.bilibili import BiliBiliLoader
#
# loader = BiliBiliLoader(['https://www.bilibili.com/video/BV1t8411y7fp/?p=4&spm_id_from=pageDriver&vd_source=9bfa62da16aae5e7da38cd1197e6acc7'])
# loader = loader.load()
# split_docs = RecursiveText.split_documents(loader)
# print(len(split_docs))
#
# from langchain_community.document_loaders import YoutubeLoader
#
# urls = [
#     "https://www.youtube.com/watch?v=HAn9vnJy6S4",
#     "https://www.youtube.com/watch?v=dA1cHGACXCo",
#     "https://www.youtube.com/watch?v=ZcEMLz27sL4",
#     "https://www.youtube.com/watch?v=hvAPnpSfSGo",
#     "https://www.youtube.com/watch?v=EhlPDL4QrWY",
#     "https://www.youtube.com/watch?v=mmBo8nlu2j0",
#     "https://www.youtube.com/watch?v=rQdibOsL1ps",
#     "https://www.youtube.com/watch?v=28lC4fqukoc",
#     "https://www.youtube.com/watch?v=es-9MgxB-uc",
#     "https://www.youtube.com/watch?v=wLRHwKuKvOE",
#     "https://www.youtube.com/watch?v=ObIltMaRJvY",
#     "https://www.youtube.com/watch?v=DjuXACWYkkU",
#     "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
# ]
# docs = []
# for url in urls:
#     docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())



"""
    索引文档
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)
embeddings = DashScopeEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)

pass