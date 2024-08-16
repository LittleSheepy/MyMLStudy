from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
# 提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个有用的助手。尽你所能回答所有问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 历史会话存储
store = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 Tongyi LLM，并设置温度为 1，代表模型会更加随机，但也会更加不确定
llm = Tongyi(temperature=1)

# 构建链式调用
chain = prompt | llm

# 历史消息
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# 对话1
config = {"configurable": {"session_id": "abc2"}}

# 将一个消息列表传递给 .invoke 方法
response1 = with_message_history.invoke({"messages": [HumanMessage(content="你好，我是小明")]}, config=config)
print(response1)

# 模型有历史聊天记录，再次提问
response2 = with_message_history.invoke({"messages":[HumanMessage(content="我的姓名是什么?")]}, config=config)
print(response2)

from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# k=10 则无法记得姓名是什么，k=20 则可以记得
def filter_messages(messages, k=10):
    return messages[-k:]


# 提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个有用的助手。尽你所能用{language}回答所有问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 历史会话存储
store = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 Tongyi LLM，并设置温度为 1，代表模型会更加随机，但也会更加不确定
llm = Tongyi(temperature=1)

# 构建链式调用
chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | llm
)

# 历史消息
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# 对话1
config = {"configurable": {"session_id": "abc3"}}

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

# 模型有历史聊天记录，再次提问
response1 = with_message_history.invoke(
    {"messages": messagesList + [HumanMessage(content="我的姓名是什么?")], "language":"Chinese"},
    config=config
)
print(response1)

response1 = with_message_history.invoke(
    {"messages": messagesList + [HumanMessage(content="我喜欢的冰淇淋是什么?")], "language":"Chinese"},
    config=config
)
print(response1)

config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r, end="|")
