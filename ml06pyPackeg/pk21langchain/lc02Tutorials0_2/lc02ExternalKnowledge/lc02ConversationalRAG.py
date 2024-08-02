# Retrieval Augmented Generation 检索增强生成
import os

import bs4
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

llm = Tongyi()
"""
    创建检索器
"""
# 1.加载 Load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 2.拆分 chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3.储存
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# 2. Incorporate the retriever into a question-answering chain.
"""
你是回答问题的助理
使用以下检索到的上下文来回答问题
如果你不知道答案，就说你不知道
最多使用三句话，并保持答案简明扼要
"""
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What is Task Decomposition?"})
print(response["answer"])

"""
提供聊天历史记录和最新的用户问题
可能参考聊天历史中的上下文，形成一个独立的问题
这可以在没有聊天历史的情况下理解。不要回答问题
如果需要，只需重新制定，否则就按原样返回。
"""
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

#对话模版 添加了系统和真人
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),      # 加载之前的系统模版
        MessagesPlaceholder("chat_history"),            # 被用来指定一个变量名为 "chat_history"，这个变量名将在后续的对话中用来存储历史聊天消息
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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# 这里附带了历史的设定
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

"""
    开始对话
"""
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

question = "What is Task Decomposition?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_2["answer"]])

print(ai_msg_2["answer"])

for document in ai_msg_2["context"]:
    print(document)
    print()



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]

conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")






























