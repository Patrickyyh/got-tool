
from dotenv import load_dotenv
## For embedding the document
from langchain.embeddings import OpenAIEmbeddings
## import Chroma DB to store the vectors.
from langchain.vectorstores.chroma import Chroma
## For RetrievalQA
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


from redundant_fileter_retriever import RedudantFilterRetriever
import  langchain
langchain.debug = True

load_dotenv()

embeddings = OpenAIEmbeddings()

## Load from disk
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

## Use the customized retriever
retriever = RedudantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chat = ChatOpenAI(verbose=True)
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about English language")
print(result)
