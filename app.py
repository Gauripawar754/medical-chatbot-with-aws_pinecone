from flask import Flask,render_template,jsonify, request

from src.helper import download_model,load_pdf_file, split_data
from src.prompt import *

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
import os


app = Flask(__name__)





PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embedding =download_model()


index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)


retriever = docsearch.as_retriever(search_type ="similarity", search_kwargs={"k":3})

chatmodel = ChatOpenAI(model = "gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ])


question_answer_chain =create_stuff_documents_chain(chatmodel,prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)




@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input":msg})
    return str(response["answer"])

if __name__ =="__main__":
    app.run(debug=True)