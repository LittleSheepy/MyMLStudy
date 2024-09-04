import os
import langchain
print(langchain.__version__)

# os.environ["MOONSHOT_API_KEY"] = ""
# os.environ["DASHSCOPE_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = "https://openkey.cloud/v1"

def MyOpenAI_chat():
    from langchain_openai import ChatOpenAI
    llm_chat = ChatOpenAI()
    ret = llm_chat.invoke("1+1等于几")
    print(ret.content)
def MyOpenAI():
    from langchain_openai import OpenAI
    llm_chat = OpenAI()
    ret = llm_chat.invoke("1+1等于几")
    print(ret)

def MyTongyi_chat():
    from langchain_community.chat_models import ChatTongyi
    llm_chat = ChatTongyi(model="qwen-max")
    ret = llm_chat.invoke("1+1等于几")
    print(ret.content)

def MyTongyi():
    from langchain_community.llms import Tongyi
    llm_chat = Tongyi()
    ret = llm_chat.invoke("1+1等于几")
    print(ret)


def MyMoonshotChat():
    from langchain_community.chat_models.moonshot import MoonshotChat
    llm_chat = MoonshotChat()
    ret = llm_chat.invoke("1+1等于几")
    print(ret.content)


if __name__ == '__main__':
    # MyOpenAI_chat()
    # MyOpenAI()
    # MyTongyi_chat()
    MyTongyi()
    # MyMoonshotChat()

