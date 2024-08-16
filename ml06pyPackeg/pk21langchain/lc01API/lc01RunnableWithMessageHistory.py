import os
from operator import itemgetter

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
model = ChatTongyi(model="qwen-max")
"""
RunnableWithMessageHistory，可用于任何的chain中添加对话历史，将以下之一作为输入
（1）一个BaseMessage序列
（2）一个dict，其中一个键的值是一个BaseMessage序列
（3）一个dict，其中一个键的值存储最后一次对话信息，另外一个键的值存储之前的历史对话信息

输出以下之一
（1）一个可以作为AIMessage的content的字符串
（2）一个BaseMessage序列
（3）一个dict，其中一个键的值是一个BaseMessage序列
"""

store = {}

def get_session_history(session_id):#一轮对话的内容只存储在一个key/session_id
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

"""
    case 1. Messages input, message(s) output
    给ChatModel（ChatTongyi）添加memory，ChatModels接受一个消息列表作为输入，并输出一条消息。
"""
print("-"*40,"\n","         case 1. Messages input, message(s) output\n","-"*40,"\n")
with_message_history = RunnableWithMessageHistory(model, get_session_history)
result = with_message_history.invoke(input=HumanMessage("你好，我是小明。") ,config={'configurable':{'session_id':'id123'}})
print(result.content)   # AIMessage
result = with_message_history.invoke(input=HumanMessage("我的名字是什么？") ,config={'configurable':{'session_id':'id123'}})
print(result.content)   # AIMessage

"""
    case 2. Dictionary input, message(s) output
    包装 prompt + LLM.因为prompt的输入是dict，所以这里输入为dict。
"""
print("-"*40,"\n","         case 2. Dictionary input, message(s) output\n","-"*40,"\n")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个助手，擅长能力{ability}。用20个字以内回答"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
i1=with_message_history.invoke(
    {"ability": "数学", "input": HumanMessage("什么是余弦定理")},
    config={"configurable": {"session_id": "abc123"}},#历史信息存入session_id
)
print(i1.content)
i2=with_message_history.invoke(
    {"ability": "math", "input": HumanMessage("重新回答一次")},
    config={"configurable": {"session_id": "abc123"}},#历史信息存入session_id
)
print(i2.content)#记忆到了

"""
    case 3. Dict with single key for all messages input, messages output
"""
print("-"*40,"\n","         case 3. Dict with single key for all messages input, messages output\n","-"*40,"\n")
runnable_with_history = RunnableWithMessageHistory(
    itemgetter("input_messages") | model,
    get_session_history,
    input_messages_key="input_messages",
)
result = runnable_with_history.invoke(
    {"input_messages": [HumanMessage(content="hi - im bob!")]},
    config={"configurable": {"session_id": "4"}},
)
print(result.content)   # AIMessage

result = runnable_with_history.invoke(
    {"input_messages": [HumanMessage(content="whats my name?")]},
    config={"configurable": {"session_id": "4"}},
)
print(result.content)   # AIMessage


"""
    case 4. Messages input, dict output
"""
print("-"*40,"\n","         case 4. Messages input, dict output\n","-"*40,"\n")
chain = RunnableParallel({"output_message": model})
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_message",
)

i31 = with_message_history.invoke(
    [HumanMessage(content="白雪公主是哪里的人物？")],
    config={"configurable": {"session_id": "baz"}},
)
print(i31["output_message"].content)

i31 = with_message_history.invoke(
    [HumanMessage(content="请简单介绍一下她。")],
    config={"configurable": {"session_id": "baz"}},
)
print(i31["output_message"].content)






pass






