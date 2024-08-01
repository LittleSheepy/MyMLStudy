import os
import langchain
from langchain_community.llms import Tongyi

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
print(langchain.__version__)
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

model = Tongyi()

# result = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# print(result)

# result = model.invoke([HumanMessage(content="What's my name?")])
# print(result)


#
# result = model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )

# print(result)


from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是一个有用的助手。用{language}尽你所能回答所有问题。"),
        # ("system","你是一个有用的助手。尽你所能回答所有问题。",),
        # MessagesPlaceholder(variable_name="messages"),
        ("user", "{messages}"),
    ]
)

# 构建链式调用
chain = prompt | model
# with_message_history = RunnableWithMessageHistory(model, get_session_history)
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc2"}}
#
# messages = [
#     SystemMessage(content="你是一个有用的助手。尽你所能回答所有问题。"),
#     HumanMessage(content="你好，我是小明"),
# ]
# response = with_message_history.invoke(
#     messages,
#     config=config,
# )
#
# print(response)
#
# messages = [
#     SystemMessage(content="你是一个有用的助手。尽你所能回答所有问题。"),
#     HumanMessage(content="我的姓名是什么?"),
# ]
# response = with_message_history.invoke(
#     messages,
#     config=config,
# )
#
# print(response)
#
# config = {"configurable": {"session_id": "abc3"}}
#
# response = with_message_history.invoke(
#     [HumanMessage(content="我的姓名是什么?")],
#     config=config,
# )
# print(response)
#
#
# response = chain.invoke(
#     {"messages": [HumanMessage(content="你好，我是小明")], "language": "Chinese"}
# )
#
# print(response)
#


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "abc11"}}
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="你好，我是小红")], "language": "Spanish"},
    config=config,
)

print(response)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="我的姓名是什么?")], "language": "Spanish"},
    config=config,
)

print(response)

from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# 对话1
config = {"configurable": {"session_id": "abc3"}}
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
messagesList = [
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
# 报错
# response = trimmer.invoke(messages)
# print(response)

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough
# k=10 则无法记得姓名是什么，k=20 则可以记得
def filter_messages(messages, k=10):
    return messages[-k:]


# 构建链式调用
# chain = (
#     RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
#     | prompt
#     | model
# )
chain = (
    # RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) |
    prompt |
    model
)
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    },
    config=config
)
print(response)


config = {"configurable": {"session_id": "abc15"}}
for r in chain.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r, end="|")


















