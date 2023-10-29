
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    MessagesPlaceholder ,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

from langchain.agents import OpenAIFunctionsAgent , AgentExecutor
from langchain.schema import SystemMessage


from dotenv import load_dotenv
load_dotenv()


from tools.sql import run_query_tool , list_tables, describe_tables_tool


chat = ChatOpenAI()
tables = list_tables()
print(tables)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = (f"You are an AI that has access to a SQLite Database. \n"
                                 f"The database has tables of: {tables} \n"
                                 "Do not make any assumption about what tables exist "
                                 "or what columns exist in each table. Instead, use the 'describe_tables' function"
                    )),
        HumanMessagePromptTemplate.from_template("{input}"),
        ## Consider the scratchpad as a very simple version of memory
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool , describe_tables_tool]
agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools= tools
)

agent_executor = AgentExecutor(
    agent= agent,
    verbose=True,
    tools = tools
)


agent_executor("How many users have provided a shipping address?")


