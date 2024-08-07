# Summarize Text 总结文本
import os

from langchain import hub
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

llm = ChatTongyi(model="qwen-max")

"""
    快速入门
"""
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# loader = WebBaseLoader("https://zhuanlan.zhihu.com/p/668153266")
# docs = loader.load()
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

file_content = read_txt_file('txt.txt')
docs = [Document(page_content=file_content)]

chain = load_summarize_chain(llm, chain_type="stuff")

# result = chain.invoke(docs)
# print(result["output_text"])
# result = chain.run(docs)
# print(result)


"""
    选项 1. Stuff
"""
# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
# result = stuff_chain.invoke(docs)
# print(result["output_text"])


"""
    选项 2. Map-Reduce
"""

""" 创建prompt  创建提示 """
# Map
# 需要设置LANGCHAIN_API_KEY
# map_prompt = hub.pull("rlm/map-prompt")

map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)


map_prompt = ChatPromptTemplate.from_messages([
  ("human", """The following is a set of documents:
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""),
])

map_chain = LLMChain(llm=llm, prompt=map_prompt)


# Reduce
# 需要设置LANGCHAIN_API_KEY
# reduce_prompt = hub.pull("rlm/reduce-prompt")

reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


# 获取一个文档列表，将它们组合成一个字符串，并将其传递给LLMChain
combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")

# Combines and iteratively reduces the mapped documents 合并并迭代地缩减映射文档
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called. 这是最后一个链。
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain` 如果文档超出了`StuffDocumentsChain`的上下文
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into. 将文档分组到的最大标记数。
    token_max=4000,
)

# Combining documents by mapping a chain over them, then combining results 通过在文档上映射一个链来组合文档，然后组合结果
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in llm_chain中存放文档的变量名
    document_variable_name="docs",
    # Return the results of the map steps in the output 在输出中返回map步骤的结果
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

result = map_reduce_chain.invoke(split_docs)

print(result["output_text"])


"""
    选项 3.Refine
"""
chain = load_summarize_chain(llm, chain_type="refine")
result = chain.invoke(split_docs)

print(result["output_text"])


prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": split_docs}, return_only_outputs=True)

prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
print(result["output_text"])
print("\n\n".join(result["intermediate_steps"][:3]))

from langchain.chains import AnalyzeDocumentChain

summarize_document_chain = AnalyzeDocumentChain(
    combine_docs_chain=chain, text_splitter=text_splitter
)
summarize_document_chain.invoke(docs[0].page_content)
















