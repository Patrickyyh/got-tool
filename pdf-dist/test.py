from typing import Any, Dict, Iterator, Optional, Union
from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable.config import RunnableConfig
from queue import Queue
from threading import Thread

load_dotenv()


#
# queue = Queue()
class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs: Any) -> Any:
        self.queue.put(None)


chat = ChatOpenAI(model="gpt-4-1106-preview", streaming=True)
prompt = ChatPromptTemplate.from_messages([("human", "{content}")])


class StreamableChain(LLMChain):
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            self(input, callbacks=[handler])

        Thread(target=task).start()
        while True:
            if not queue.empty():
                token = queue.get()
                if token is None:
                    break
                yield token


class StreamingChain(StreamableChain, LLMChain):
    pass


## Queue object is thread safe in python.
chain = StreamingChain(llm=chat, prompt=prompt)
for output in chain.stream(input={"content": "please tell me a joke"}):
    print(output)
