
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    MessagesPlaceholder ,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

from langchain.agents import OpenAIFunctionsAgent , AgentExecutor
from dotenv import load_dotenv
load_dotenv()


from tools.sql import run_query_tool

chat = ChatOpenAI()
prompt = ChatPromptTemplate(
    input_variables=["input"],
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        ## Consider the scratchpad as a very simple version of memory
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools= [run_query_tool]
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools = [run_query_tool]
)


agent_executor("How many users are in the database")


