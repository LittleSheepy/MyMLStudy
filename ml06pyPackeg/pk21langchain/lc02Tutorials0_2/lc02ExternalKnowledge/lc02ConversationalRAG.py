# Conversational RAG 会话检索增强生成
import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import chat_agent_executor
from langgraph.checkpoint.sqlite import SqliteSaver

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
llm = ChatTongyi(model="qwen-max")

"""
    创建检索器
"""
print("-"*40,"\n","         创建检索器\n","-"*40,"\n")
# 1.加载 Load
loader = TextLoader(r'txt.txt', encoding='utf8')
docs = loader.load()

# 2.拆分 chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3.储存
vectorstore = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings())
retriever = vectorstore.as_retriever()


"""
    Chains 链
"""
print("-"*40,"\n","         Chains 链\n","-"*40,"\n")
# 2. Incorporate the retriever into a question-answering chain.   将检索器合并到问答链中。
system_prompt = """
你是回答问题的助理
使用以下检索到的上下文来回答问题
如果你不知道答案，就说你不知道
最多使用三句话，并保持答案简明扼要
"\n\n"
"{context}"
"""
"""
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# create_stuff_documents_chain 它接受检索到的上下文以及会话历史和查询，以生成答案。
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "什么是任务分解?"})
print(response["answer"])

"""
    Adding chat history 添加聊天记录
"""
print("-"*40,"\n","         Adding chat history 添加聊天记录\n","-"*40,"\n")

contextualize_q_system_prompt = """
提供聊天历史记录和最新的用户问题
可能参考聊天历史中的上下文，形成一个独立的问题
这可以在没有聊天历史的情况下理解。不要回答问题
如果需要，只需重新制定，否则就按原样返回。
"""
"""
"Given a chat history and the latest user question "
"which might reference context in the chat history, "
"formulate a standalone question which can be understood "
"without the chat history. Do NOT answer the question, "
"just reformulate it if needed and otherwise return it as is."
"""

#对话模版 添加了系统和真人
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # 加载之前的系统模版
        MessagesPlaceholder("chat_history"),        # 被用来指定一个变量名为 "chat_history"，这个变量名将在后续的对话中用来存储历史聊天消息
        ("human", "{input}"),
    ]
)

#创建一个历史对话系统 加载 大模型，向量数据库，历史对话模版
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 它接受检索到的上下文以及会话历史和查询，以生成答案。
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)    # 其中输入键为context，，chat_history和input
# 这里附带了历史的设定
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

"""
    开始对话
"""
print("-"*40,"\n","         开始对话\n","-"*40,"\n")
chat_history = []
question = "什么是任务分解?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)
second_question = "通常的做法是什么?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_2["answer"]])
print(ai_msg_2["answer"])
#检查引用的内容
for document in ai_msg_2["context"]:
    print(document)
    print("--------------------------------------")

"""
    第二种方案
"""
print("-"*40,"\n","         第二种方案\n","-"*40,"\n")
# 加载历史信息
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

result = conversational_rag_chain.invoke(
    {"input": "什么是任务分解?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
print(result)

result = conversational_rag_chain.invoke(
    {"input": "通常的做法是什么?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]
print(result)

for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")

"""
    Agents
"""
print("-"*40,"\n","         Agents\n","-"*40,"\n")
# 1.Retrieval tool 检索工具
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",  # name
    "Searches and returns excerpts from the Autonomous Agents blog post.",  # description
)
tools = [tool]

result = tool.invoke("任务分解")
print(result)

# 2.Agent constructor 代理建造者
agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)

# query = "什么是任务分解?"
query = "What is Task Decomposition?"
for s in agent_executor.stream({"messages": [HumanMessage(content=query)]}):
    # print(type(s["agent"]["messages"][0]))      # AIMessage
    print(s)
    print("-----------------")

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

for s in agent_executor.stream({"messages": [HumanMessage(content="你好，我是小明。")]}, config=config):
    # print(s["agent"]["messages"][0].content)
    print(type(s["agent"]["messages"][0]))
    print(s["agent"]["messages"][0].content)
    print("-----------------")

# query = "什么是任务分解？"
query = "What is Task Decomposition?"
for s in agent_executor.stream({"messages": [HumanMessage(content=query)]}, config=config):
    print(s)
    print("----")

# query = "根据这篇博文，常见的做法是什么?重新搜索"
query = "What according to the blog post are common ways of doing it? redo the search"
for s in agent_executor.stream({"messages": [HumanMessage(content=query)]}, config=config):
    print(s)
    print("----")


pass



