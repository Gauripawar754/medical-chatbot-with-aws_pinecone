from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document



# Extract data from pdf files
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    document = loader.load()
    return document


def filter_minimal_doc(docs: List[Document])  -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_doc : List[Document] =[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_doc.append(
            Document(
                page_content= doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_doc
    



# split the data into smaller chunks
def split_data(extracted_data):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)
    text_chunks = text_spliter.split_documents(extracted_data)
    return text_chunks


#download  embedding model
def download_model():
 embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
 return embedding