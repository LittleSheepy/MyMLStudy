import os
import langchain
from langchain_community.llms import Tongyi
print(langchain.__version__)
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
llm_tongyi = Tongyi()
print(llm_tongyi.invoke("1+1等于几"))
# print(llm_tongyi.invoke("What is 13 raised to the .3432 power?"))