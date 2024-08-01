import os

from langchain_community.llms import Tongyi

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"


model = Tongyi()




from langchain_community.tools.tavily_search import TavilySearchResults

# search = TavilySearchResults(max_results=1)
#
# result = search.invoke("今天重庆的天气预报")
# print(result)



from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://help.aliyun.com/zh/dashscope/product-overview/concepts-and-glossary?spm=a2c4g.11186623.0.0.63955491NXmvJ5")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, DashScopeEmbeddings())
retriever = vector.as_retriever()

# result = retriever.invoke("灵积模型是什么？")
# print(result)


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "搜索灵积模型",
    "搜索灵积模型的信息，你可以使用这个工具!",
)

# 创建工具列表
# toolList = [search, retriever_tool]
toolList = [retriever_tool]



from langchain_core.tools import tool
@tool
def multiply(first_int: int, second_int: int) -> int:
    """两个数的乘积."""
    return first_int * second_int


# 创建工具列表
toolList = [retriever_tool, multiply]

from langchain_community.chat_models.tongyi import ChatTongyi
llm = ChatTongyi(model="qwen-max")
llm_with_tools = llm.bind_tools([retriever_tool, multiply])

# # 未使用工具
# msg = llm.invoke("你好")
# print(msg)
#
# # 建议使用工具：search_dashscope_knowledge
# msg = llm_with_tools.invoke("灵积模型是什么")
# print(msg)
#
# # 建议使用工具：multiply
# msg = llm_with_tools.invoke("计算 10 的 5 倍")
# print(msg)
""" 创建代理 """
# 创建一个代理
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, toolList)
msg = agent_executor.invoke(
    {"messages": [HumanMessage(content="你好")]}
)
print(msg)

# # 建议使用工具：search_dashscope_knowledge
# msg = agent_executor.invoke(
#     {"messages": [HumanMessage(content="灵积模型是什么")]}
# )
# print(msg)

# # 建议使用工具：multiply
# msg = llm_with_tools.invoke("计算 10 的 5 倍等于的结果")
# print(msg)
#
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="你好")]}
# ):print(chunk)
# print("----")
#
#
# # 建议使用工具：search_dashscope_knowledge
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="灵积模型是什么")]}
# ):print(chunk)
# print("----")


# 建议使用工具：multiply
for chunk in llm_with_tools.stream("计算 10 的 5 倍的结果"):print(chunk)
print("----")


async def main():
    async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content="你好")]}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                    event["name"] == "Agent"
            ):
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                    event["name"] == "Agent"
            ):
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

# 创建缓存
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")


# 创建代理
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
agent_executor = create_react_agent(llm, toolList, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
# agent_executor = chat_agent_executor.create_tool_calling_executor(
#     model, tools, checkpointer=memory
# )

config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="你好，我是小明")]},config=config
):print(chunk)
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="我的名字是什么？")]},config=config
):print(chunk)

import asyncio

if __name__ == "__main__":
    asyncio.run(main())

pass





