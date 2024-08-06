# Build a Local RAG Application 构建本地RAG应用
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
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
llm = Tongyi()
"""
    文档加载
"""
print("-"*40,"\n","         文档加载\n","-"*40,"\n")
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()
# 读取txt文件的示例代码
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# 使用示例
file_content = read_txt_file('txt.txt')
data = [Document(page_content=file_content)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

llm = Tongyi()


local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=DashScopeEmbeddings())

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(len(docs))            #  -> 4
print(docs[0])              #


from langchain_community.llms import LlamaCpp
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=r"D:\BaiduNetdiskDownload/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    # model_path="/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
# ggml_metal_init: allocating
# ggml_metal_init: using MPS

result = llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")

print(result)

# from langchain_community.llms import GPT4All
#
# gpt4all = GPT4All(
#     model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin",
#     max_tokens=2048,
# )

# from langchain_community.llms.llamafile import Llamafile
# llamafile = Llamafile()
# llamafile.invoke("Here is my grandmother's beloved recipe for spaghetti and meatballs:")



"""
    链式使用
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

# Run
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
chain.invoke(docs)



"""
    问答
"""
from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
print(rag_prompt.messages)
from langchain_core.runnables import RunnablePassthrough, RunnablePick

# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run
result = chain.invoke({"context": docs, "question": question})
print(result)
# Prompt
rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
rag_prompt_llama.messages
print(rag_prompt_llama.messages)
# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt_llama
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
"""
    检索问答
"""
retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

qa_chain.invoke(question)


from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)

response_message = model.invoke(
    "Simulate a rap battle between Stephen Colbert and John Oliver"     # 模拟史蒂芬·科尔伯特和约翰·奥利弗之间的说唱对决
)

print(response_message.content)













