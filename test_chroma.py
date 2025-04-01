# test_chroma.py
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

docs_split = []  # Empty list for testing
client = chromadb.Client()
vectorstore = Chroma.from_documents(
    documents=docs_split,
    collection_name="Hipertension",
    embedding=OpenAIEmbeddings(),
    client=client,
)
print("Vectorstore created successfully!")
