# Build an Extraction Chain 建立提取链
import os
from typing import List, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

"""
    单个实体
"""
class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="个人姓名")
    hair_color: Optional[str] = Field(
        default=None, description="如果已知，该人头发的颜色。"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="身高，以米为单位测量。"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专门用于信息提取的算法专家。"
            "请在处理文本时，仅提取与任务相关的信息。"
            # "如果你遇到一个无法确定其值的属性， "
            # "请在相应的属性位置返回null值。"
            ,
        ),
        ("human", "{text}"),
    ]
)

# langchain_mistralai import ChatMistralAI
# llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

llm = ChatTongyi(model="qwen-max")
structured_llm = llm.with_structured_output(schema=Person)
runnable = prompt | structured_llm

text = "我的名字叫吴磊，我有一头黑色的短发，我身高一米六五。"
result = structured_llm.invoke(text)
print(result)

result = runnable.invoke({"text": text})
print(result)

"""
    多个实体
"""

llm = ChatTongyi(model="qwen-max")
class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

structured_llm = llm.with_structured_output(schema=Data)
runnable = prompt | structured_llm
text = "我的名字叫吴磊，我有一头黑色的短发，我身高一米六五。张序言有和我的一样的头发颜色，她身高一米七三。"

result = structured_llm.invoke(text)
print(result)

result = runnable.invoke({"text": text})
print(result)




















