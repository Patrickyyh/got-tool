from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.chat.vector_stores.pinecone import vectorstore

def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """
    Generate and store embeddings for the given pdf
    1. Extract text from the specified PDF.
    2. Divide the extracted text into manageable chunks.
    3. Generate an embedding for each chunk.
    4. Persist the generated embeddings.

    :param pdf_id: The unique identifier for the PDF.
    :param pdf_path: The file path to the PDF.

    Example Usage:

    create_embeddings_for_pdf('123456', '/path/to/pdf')
    """
    ## 1. Create a TextSpliter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        is_separator_regex= False,
    )

    ## 2. Create a PDFLoader and use the splitter to split PDF into the text chunks.
    loader = PyPDFLoader(pdf_path)
    docs  = loader.load_and_split(text_splitter)
    ##print(docs)

    ## 3. Add the docs into the vector stores
    vectorstore.add_documents(docs)



    pass
