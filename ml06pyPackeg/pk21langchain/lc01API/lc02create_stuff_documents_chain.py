import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
llm = ChatTongyi(model="qwen-max")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """根据提供的上下文: {context} \n\n"""),
        ("human", """{input}"""),
    ]
)

# 构建链
chain = create_stuff_documents_chain(llm, prompt)

# 定义文档内容
docs = [
    Document(page_content="杰西喜欢红色，但不喜欢黄色"),
    Document(page_content="贾马尔喜欢绿色，有一点喜欢红色"),
    Document(page_content="玛丽喜欢粉色和红色")
]

# 执行链
res = chain.invoke({"input": "大家喜欢什么颜色?", "context": docs})
print(res)
