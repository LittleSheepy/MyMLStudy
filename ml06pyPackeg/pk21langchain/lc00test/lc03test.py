import os
import langchain
from langchain_community.llms import Tongyi

from langchain_community.llms import QianfanLLMEndpoint
print(langchain.__version__)
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"



llm_tongyi = Tongyi()
llm_qianfan = QianfanLLMEndpoint(temperature=0.1)
print(llm_qianfan.invoke("1+1等于几"))
