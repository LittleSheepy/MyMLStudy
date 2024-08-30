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
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langgraph.prebuilt import chat_agent_executor
from langgraph.checkpoint.sqlite import SqliteSaver

os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"
llm = ChatTongyi(model="qwen-max")

txt = """藜麦(18张)藜麦是印第安人的传统主食，几乎和水稻同时被驯服有着600备才纹0多年的种植和食用历史。藜麦具有相当全面营养成分，并且迁踏埋藜麦的口感口味都容易被人接受。在藜麦这种营洒樱养丰富的粮食滋养下精探南美洲的印第安人创造了伟大的印加文明，印加人将藜麦尊为粮食之母。美国人早在80年代就将多臭棵尝藜麦引跨拜讲入NASA，作为宇航员的日常口粮润棵，FAO认定藜麦是唯一一种单作物即可满足人类所需的全部营养的粮食，并进行藜麦的推广和宣传。2013年是联合国钦定的国际蒸企胶藜麦年。以此呼吁人们注意粮食安全和营养均衡。 [1]\n藜麦穗部可呈红、紫、黄，植株形状类似灰灰菜，成熟后穗部类似高粱穗。植株大小受环境及遗传因素影响较大，从0.3-3米不等，茎部质地较硬，可分枝可不分。单叶互生，叶片呈鸭掌状，叶缘分为全缘型与锯齿缘型。根系庞大但分布较浅，根上的须根多，吸水能力强。藜麦花两性，花序呈伞状、穗状、圆锥状，藜麦种子较小，呈小圆药片状，直径1.5-2毫米，千粒重1.4-3克。 [1]\n原产于南美洲安第斯山脉的哥伦比亚、厄瓜多尔、秘鲁等中高海拔山区。具有一定的耐旱、耐寒、耐盐性，生长范围约为海平面到海拔4500米左右的高原上，最适的高度为海拔3000-4000米的高原或山地地区。 [1] [4]\n繁殖地块选择：应选择地势较高、阳光充足、通风条件好及肥力较好的地块种植。藜麦不宜重茬，忌连作，应合理轮作倒茬。前茬以大豆、薯类最好，其次是玉米、高粱等。 [4]\n施肥整地：早春土壤刚解冻，趁气温尚低、土壤水分蒸发慢的时候，施足底肥，达到土肥融合，壮伐蓄水。播种前每降1次雨及时耙耱1次，做到上虚下实，干旱时只耙不耕，并进行压实处理。一般每亩（667平方米/亩，下同）施腐熟农家肥1000-2000千克、硫酸钾型复合肥20-30千克。如果土壤比较贫瘠，可适当增加复合肥的施用量。 [4]\n播种期一般选在5月中旬、气温在15-20℃时为宜。播种量为每亩0.4千克。播种深度1-2厘米。一般使用耧播，也可采用谷子精量播种机播种。行距50厘米左右，株距15-25厘米。 [4]\n苗期查苗补苗：藜麦出苗后，要及时查苗，发现漏种和缺苗断垄时，应采取补种。对少数缺苗断垄处，可在幼苗4-5叶时雨后移苗补栽。移栽后，适度浇水，确保成活率。对缺苗较多的地块，采用催芽补种，先将种子浸入水中3-4小时，捞出后用湿布盖上，放在20-25℃条件下闷种10小时以上，然后开沟补种。 [4]\n间苗定苗：藜麦出苗后应及早间苗，并注意拔除杂草。当幼苗长到10厘米、长出5-6叶时间苗，按照留大去小的原则，株距保持在15-25厘米。 [4]\n中耕除草：中耕结合间苗进行，应掌握浅锄、细锄、破碎土块，围正幼苗，做到深浅一致，草净地平，防止伤苗压苗。中耕后如遇大雨，应在雨后表土稍干时破除板结。当藜麦长到50厘米以上时，还需除草1-2次。 [4]\n在藜麦8叶龄时，将行中杂草、病株及残株拔掉，提高整齐度，增加通风透光，同时，进行根部培土，防止后期倒伏。 [4]\n要求一次性施足底肥，如果生长中后期发现有缺肥症状，可适当追肥。一般在植株长到40-50厘米时，每亩撒施三元复合肥10千克。在藜麦生育后期，叶面喷洒磷肥和微量元素肥料，可促进开花结实和子粒灌浆。藜麦主要以旱作为主，如发生严重干旱，应及时浇水。 [4]\n在成熟期，要严防麻雀为害，及时收获，防止大风导致脱粒，造成损失。 [4]\n叶斑病病害：主要防治叶斑病，使用12.5%的烯唑醇可湿性粉剂3000-4000倍液喷雾防治，一般防治1-2次即可收到效果。 [4]\n虫害：藜麦常见虫害有象甲虫、金针虫、蝼蛄、黄条跳甲、横纹菜蝽、萹蓄齿胫叶甲、潜叶蝇、蚜虫、夜蛾等。防治方法：可每亩用3%的辛硫磷颗粒剂2-2.5千克于耕地前均匀撒施，随耕地翻入土中。也可以每亩用40%的辛硫磷乳油250毫升，加水1-2千克，拌细土20-25千克配成毒土，撒施地面翻入土中，防治地下害虫。 [4] [7]\n藜麦藜麦的营养价值超过任何一种传统的粮食作物，藜麦是一种全谷全营养完全蛋白碱性食物，藜麦作为一种藜科植物其蛋白质含量与牛肉相当，其品质也不亚于肉源蛋白与奶源蛋白。藜麦所含氨基酸种类丰富，除了人类必须的9种必须氨基酸，还含有许多非必须氨基酸，特别是富集多数作物没有的赖氨酸，并且含有种类丰富且含量较高的矿物元素，以及多种人体正常代谢所需要的维生素，不含胆固醇与麸质，糖含量、脂肪含量与热量都属于较低水平。 [1]\n藜麦富含的维生素、多酚、类黄酮类、皂苷和植物甾醇类物质具有多种健康功效。藜麦具有高蛋白，其所含脂肪中不饱和脂肪酸占83%。藜麦还是一种低果糖低葡萄糖的食物，能在糖脂代谢过程中发挥有益功效。 [5]\n藜麦的全营养性和高膳食纤维等特性决定了它对健康的益处。研究表明，藜麦富含的维生素、多酚、类黄酮类、皂苷和植物甾醇类物质具有多种健康功效。藜麦具有高蛋白，其所含脂肪中不饱和脂肪酸占83%，还是一种低果糖低葡萄糖的食物能在糖脂代谢过程中发挥有益功效。 [1]\n
"""
"""
    创建检索器
"""
print("-"*40,"\n","         创建检索器\n","-"*40,"\n")
# 1.加载 Load
loader = TextLoader(r'藜.txt', encoding='utf8')
docs = loader.load()

# 2.拆分 chunk  Recursive
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=128,
    chunk_overlap=0,
    length_function=len,
)
#
splits = text_splitter.create_documents([docs[0].page_content],metadatas=[docs[0].metadata])
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
# splits = text_splitter.split_documents(docs)

# 3.储存
vectorstore = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings())
retriever = vectorstore.as_retriever()
result = vectorstore.similarity_search("藜的播种期？")
print(result)
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

response = rag_chain.invoke({"input": "藜怎么防治虫害?"})
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



