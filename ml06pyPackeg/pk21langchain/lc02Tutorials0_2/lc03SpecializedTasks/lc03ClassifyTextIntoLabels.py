# Classify Text into Labels 将文本分类为标签
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
"""
从下面的文章中提取所需的信息。
只提取` Classification `函数中提到的属性。

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="文本的情感")
    aggressiveness: int = Field(
        description="评分1到10，这段文字有多激进。"
    )
    language: str = Field(description="文本所使用的语言")

# LLM
llm = ChatTongyi(model="qwen-max").with_structured_output(schema=Classification)

tagging_chain = tagging_prompt | llm
# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"  # 我很高兴认识你!我想我们会成为很好的朋友!
inp = "我很高兴认识你!我想我们会成为很好的朋友!"  # 我很高兴认识你!我想我们会成为很好的朋友!
res = llm.invoke(inp)
print(res)
res = tagging_chain.invoke({"input": inp})
print(res)

# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"        # 我很生你的气!我要给你你应得的。
inp = "我很生你的气!我要给你你应得的！"        # 我很生你的气!我要给你你应得的。
res = llm.invoke(inp)
print(res)

res = tagging_chain.invoke({"input": inp})
print(res)

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["开心", "中性", "伤心"])
    aggressiveness: int = Field(
        ...,
        description="描述语句的激进性，数字越高越强e",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["西班牙语", "英语", "法语", "德语", "中文"]
    )

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatTongyi(model="qwen-max").with_structured_output(schema=Classification)

chain = tagging_prompt | llm

# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"  # 我很高兴认识你!我想我们会成为很好的朋友!
inp = "我很高兴认识你!我想我们会成为很好的朋友!"  # 我很高兴认识你!我想我们会成为很好的朋友!

res = llm.invoke(inp)
print(res)

result = chain.invoke({"input": inp})
print(result)

# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"        # 我很生你的气!我要给你你应得的。
inp = "我很生你的气!我要给你你应得的！"        # 我很生你的气!我要给你你应得的。

res = llm.invoke(inp)
print(res)

result = chain.invoke({"input": inp})
print(result)

Classification(sentiment='sad', aggressiveness=5, language='spanish')

# inp = "Weather is ok here, I can go outside without much more than a coat"  # “这里的天气很好，除了一件外套我就可以出门了。”
inp = "这里的天气很好，除了一件外套我就可以出门了。"  # “这里的天气很好，除了一件外套我就可以出门了。”

res = llm.invoke(inp)
print(res)

result = chain.invoke({"input": inp})
print(result)




