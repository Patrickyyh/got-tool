from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain , SequentialChain
from langchain.prompts import MessagesPlaceholder , HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory , FileChatMessageHistory, ConversationSummaryMemory

from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(verbose=True)

## In additon to the Content variable, we also need to pass the messages variable
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages" ,
    return_messages= True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content" , "messages"],
    messages = [
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory=memory,
    verbose=True,
)


while True:
    content = input("You >> : ")
    result = chain({"content": content})
    print(result["text"])
