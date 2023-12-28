
from typing import Any, Optional, Union
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
load_dotenv()


#
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(f"from the customer handler: token :{token}")

chat = ChatOpenAI(model="gpt-4-1106-preview" ,streaming = True , callbacks=[StreamingHandler()])
prompt = ChatPromptTemplate.from_messages([
    ("human" ,"{content}")
])
chain = LLMChain(llm = chat ,  prompt=prompt)
output = chain("tell me a joke")
print(output)



# messages = prompt.format_messages(content = "tell me a joke")
# print(messages)
# for message  in chat.stream(messages):
#     print(message.content)


