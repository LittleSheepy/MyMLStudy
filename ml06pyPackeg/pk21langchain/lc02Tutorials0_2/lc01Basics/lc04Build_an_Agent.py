import os

from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
model = ChatTongyi(model="qwen-max")
os.environ["TAVILY_API_KEY"]="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"


"""
    End-to-end agent
"""
print("-"*40, "\n    End-to-end agent\n", "-"*40)
# Create the agent
memory = MemorySaver()
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("1----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("2----")

"""
    使用Tavily 搜索引擎
"""
print("-"*40, "\n    使用Tavily 搜索引擎\n", "-"*40)
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=1)

# result = search.invoke("今天上海的天气预报")
# print(result)

tools = [search]


"""
    使用语言模型
"""
print("-"*40, "\n    使用语言模型\n", "-"*40)
response = model.invoke([HumanMessage(content="你好!")])
print(response.content)


# model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke([HumanMessage(content="你好!")])
# print(f"ContentString: {response.content}")     # ContentString: 你好！有什么可以帮助你的吗？
# print(f"ToolCalls: {response.tool_calls}")      # ToolCalls: []
#
# # 让我们尝试使用一些需要调用工具的输入来调用它。
# response = model_with_tools.invoke([HumanMessage(content="What's the weather in Beijin?")])
# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")
"""
ContentString: 
ToolCalls: [{'name': 'tavily_search_results_json', 
             'args': {'query': 'weather in Beijing'}, 'id': 'call_122503b65b9040a686a891', 'type': 'tool_call'}]
"""

"""
    Create the agent 创建代理
"""
print("-"*40, "\n    Create the agent 创建代理\n", "-"*40)
agent_executor = create_react_agent(model, tools)

# 不需要工具时候
# response = agent_executor.invoke({"messages": [HumanMessage(content="你好")]})
# print(response["messages"])

# 需要工具时候
# response = agent_executor.invoke(
#     {"messages": [HumanMessage(content="whats the weather in Beijin?")]}
# )
# print(response["messages"])


"""
    Streaming Messages 流式消息
"""
print("-"*40, "\n    Streaming Messages 流式消息\n", "-"*40)
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in Beijin?")]}
# ):
#     print(chunk)
#     print("----")

"""
    Streaming tokens 流式tokens
"""
print("-"*40, "\n    Streaming tokens 流式tokens\n", "-"*40)
async def Streaming_Tokens():
    async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1"
    ):
        kind = event["event"]
        # print("event : ", kind)
        if kind == "on_chain_start":
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end=" | ")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

# asyncio.run(Streaming_Tokens())

"""
    Adding in memory 添加内存
"""
print("-"*40, "\n    Adding in memory 添加内存\n", "-"*40)
memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("1----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("2----")

config = {"configurable": {"thread_id": "xyz123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("3----")


















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





