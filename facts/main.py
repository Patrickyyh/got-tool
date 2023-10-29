from dotenv import load_dotenv

## For loading the document
from langchain.document_loaders import TextLoader

## For chucnkify the document
from langchain.text_splitter import CharacterTextSplitter

## For embedding the document
from langchain.embeddings import OpenAIEmbeddings

## import Chroma DB to store the vectors.
from langchain.vectorstores.chroma import Chroma

load_dotenv()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 200,
    chunk_overlap = 0
)


## Create the embeddings
embeddings = OpenAIEmbeddings()

## load the document
loader = TextLoader("facts.txt")

## load the document and split it into chunks
docs = loader.load_and_split(
    text_splitter = text_splitter,
)

## create Chroma DB and save the vectros in the emb directory
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory= "emb"
)

results = db.similarity_search_with_score("What is an interesting fact about English language")
for result in results:
    print("\n")
    print(f" score: {result[1]}")
    print( result[0].page_content)


# print(docs)
