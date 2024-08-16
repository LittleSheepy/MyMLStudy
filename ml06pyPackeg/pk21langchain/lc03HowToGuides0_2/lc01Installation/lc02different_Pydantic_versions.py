import os
from typing import Optional

import langchain_core
from langchain_community.chat_models import ChatTongyi
# from pydantic.v1 import BaseModel  # <-- Note v1 namespace   langchain-core<0.2.23
from pydantic import BaseModel, Field  # langchain-core>=0.2.23

print(langchain_core.__version__)   # 0.2.28
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
model = ChatTongyi(model="qwen-max")

class Person(BaseModel):
    """Personal information"""
    # name: Optional[str] = Field(default=None, description="个人姓名")
    # hair_color: Optional[str] = Field(default=None, description="如果已知，该人头发的颜色。")
    # height_in_meters: Optional[str] = Field(default=None, description="身高，以米为单位测量。")
    name: str = Field(default=None, description="姓名")
    hair_color: Optional[str]

model = model.with_structured_output(schema=Person)

text = "大家好，我的名字叫吴磊，我有一头黑色的短发，我身高一米六五。"
result = model.invoke(text)
print(result)

pass