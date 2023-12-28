
from typing import Any, Dict, Iterator, Optional, Union
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable.config import RunnableConfig
from queue import  Queue
from threading import Thread
load_dotenv()


#
queue = Queue()
class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        queue.put(token)

    def on_llm_end(self, response, **kwargs):
        queue.put(None)

    def on_llm_error(self,error, **kwargs: Any) -> Any:
        queue.put(None)


chat = ChatOpenAI(model="gpt-4-1106-preview" ,streaming = True , callbacks=[StreamingHandler()])
prompt = ChatPromptTemplate.from_messages([
    ("human" ,"{content}")
])

class StreamingChain(LLMChain):
    def stream(self ,input):
        def task():
            self(input)
        Thread(target=task).start()
        while True:
            if not queue.empty():
                token = queue.get()
                if token is None:
                    break
                yield token



## Queue object is thread safe in python.
chain = StreamingChain(llm = chat , prompt=prompt )
for output in  chain.stream(input = {"content" : "When to use asio in boost of c++"}):
    print(output)
