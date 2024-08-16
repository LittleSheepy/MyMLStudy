# from langchain_groq import ChatGroq
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableSequence
# from dotenv import load_dotenv
#
# load_dotenv()
#
# llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
#
# class AnswerGrader(BaseModel):
#     binary_score: str = Field(
#         description="Answer addresses the question, 'yes' or 'no'"
#     )
#
# structured_llm_grader = llm.with_structured_output(AnswerGrader)
#
# system = """
# You are a grader assessing whether an answer addresses / resolves a question.
# Give a binary score 'yes' or 'no', where 'yes' means that the answer resolves the question.
# """
#
# answer_prompt = ChatPromptTemplate.from_messages([
#     ("system", system),
#     ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
# ])
#
# answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
#
# if __name__ == "__main__":
#     generated = 'The different values present in the human heart can be grouped into four broad categories: cellular factors, cardiac factors, extracardiac factors, and physical factors. These values affect the transmission of the cardiac electrical field throughout the body.'
#     res = answer_grader.invoke({"question": "What causes cardiac arrest?", "generation": generated})
#     print(res)
import os
os.environ["OPENAI_API_KEY"] = "sk-kjMCtBsCdCGoVp3EAe60626fAf534c12A5009e7134Ae50C2"
os.environ["DASHSCOPE_API_KEY"] = "sk-720b1666b12f49c3915e4061e173ab15"

from langchain.llms import OpenAI,OpenAIChat
from langchain_openai import ChatOpenAI

from langchain_community.chat_models import ChatTongyi
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")
llm = ChatTongyi()
structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke("Tell me a joke about cats")
print(result)  # result: None