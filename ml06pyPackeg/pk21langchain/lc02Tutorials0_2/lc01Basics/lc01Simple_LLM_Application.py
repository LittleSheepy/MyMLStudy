import os

from langchain_community.llms import Tongyi

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"


model = Tongyi()


messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="hi!"),
]
responses = model.invoke(messages)
print(responses)       # 你好


parser = StrOutputParser()


result = parser.invoke(responses)
print(result)       # 你好


chain = model | parser
responses_chain = chain.invoke(messages)
print(responses_chain)       # 你好

""" Prompt Templates 提示模版"""
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
result = prompt_template.invoke({"language": "Chinese", "text": "hi!"})

print(type(result))
print(result)

messages1 = result.to_messages()
print(type(messages1))
print(messages1)


chain = prompt_template | model | parser
result = chain.invoke({"language": "Chinese", "text": "hi!"})
print(result)