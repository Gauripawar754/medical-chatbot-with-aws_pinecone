from dotenv import load_dotenv
load_dotenv()

import os
from src.helper import load_pdf_file, download_model, filter_minimal_doc, split_data
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec



PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extract_data = load_pdf_file(data ="data/")
filter_data = filter_minimal_doc(extract_data)
text_chunks = split_data(filter_data)

embedding  = download_model()



pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key= pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec= ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embedding,
    
)
print(index.describe_index_stats())