# Build a Chatbot 构建一个聊天机器人
import os

import langchain
from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough

print(langchain.__version__)
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

model = ChatTongyi(model="qwen-max")

result = model.invoke([HumanMessage(content="你好，我是小明")])
print(result.content)

result = model.invoke([HumanMessage(content="我的姓名是什么?")])
print(result.content)

result = model.invoke(
    [
        HumanMessage(content="你好，我是小明"),
        AIMessage(content="你好，小明！很高兴能和你交流。有什么可以帮助你的吗？"),
        HumanMessage(content="我的姓名是什么?"),
    ]
)
print(result.content)

"""
    消息历史
"""
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke([HumanMessage(content="你好，我是小明")], config=config)
print(response.content)

response = with_message_history.invoke([HumanMessage(content="我的姓名是什么?")], config=config)
print(response.content)

config2 = {"configurable": {"session_id": "abc3"}}

response = with_message_history.invoke([HumanMessage(content="我的姓名是什么?")], config=config2)
print(response.content)

# 重新用config1
response = with_message_history.invoke([HumanMessage(content="我的姓名是什么?")], config=config)
print(response.content)

"""
    提示模板
"""
print("-"*40, "\n    提示模板\n", "-"*40)
prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(content="你是一个有用的助手。用{language}尽你所能回答所有问题。"),
        ("system","你是一个有用的助手。尽你所能回答所有问题。",),
        # MessagesPlaceholder(variable_name="messages"),
        ("user", "{messages}"),
    ]
)


chain = prompt | model
response = chain.invoke({"messages": [HumanMessage(content="你好，我是小明")]})
print(response.content)

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config5 = {"configurable": {"session_id": "abc5"}}

response = with_message_history.invoke([HumanMessage(content="你好，我是小刚")], config=config5)
print(response.content)

response = with_message_history.invoke([HumanMessage(content="我的姓名是什么?")], config=config5)
print(response.content)
# 加语言
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model


response = chain.invoke({"messages": [HumanMessage(content="你好，我是小明")], "language": "Chinese"})
print(response.content)


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config11 = {"configurable": {"session_id": "abc11"}}
response = with_message_history.invoke({"messages": [HumanMessage(content="你好，我是小红")], "language": "Spanish"},config=config11)
print(response.content)

response = with_message_history.invoke({"messages": [HumanMessage(content="我的姓名是什么?")], "language": "Spanish"}, config=config11)
print(response.content)


"""
    管理对话历史记录
"""
print("-"*40, "\n    管理对话历史记录\n", "-"*40)
# k=10 则无法记得姓名是什么，k=20 则可以记得
def filter_messages(messages, k=10):
    return messages[-k:]

# 构建链式调用
chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)

messages = [
    HumanMessage(content="你好，我是小明"),
    AIMessage(content="你好!"),
    HumanMessage(content="我喜欢香草冰淇淋"),
    AIMessage(content="好的"),
    HumanMessage(content="2 + 2 等于几"),
    AIMessage(content="4"),
    HumanMessage(content="谢谢"),
    AIMessage(content="不用谢!"),
    HumanMessage(content="正玩得开心么?"),
    AIMessage(content="是的!"),
]


response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "Chinese",
    }
)
print(response.content)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my fav ice cream")],
        "language": "Chinese",
    }
)
print(response.content)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config20 = {"configurable": {"session_id": "abc20"}}
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "Chinese",
    },
    config=config20
)
print(response.content)

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my favorite ice cream?")],
        "language": "Chinese",
    },
    config=config20,
)
print(response.content)

"""
    流媒体
"""
print("-"*40, "\n    流媒体\n", "-"*40)
config15 = {"configurable": {"session_id": "abc15"}}
for r in chain.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "Chinese",
    },
    config=config15,
):
    print(r.content, end="|")


















