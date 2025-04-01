from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
import chromadb
import os


def create_vectorstore(docs_split, persist=True, persist_directory="../chroma_db"):
    """
    Create or load a persistent Chroma vector store.

    Args:
        docs_split: List of split documents to store.
        persist: Boolean to indicate if the vector store should be persisted.
        persist_directory: Directory where the vector store is saved/loaded.

    Returns:
        Chroma: The loaded or newly created vector store.
    """
    # Define the embedding function (adjust based on your setup)
    embedding_function = (
        OpenAIEmbeddings()
    )  # Replace with your embedding model if different

    # Check if the persistent directory exists and has a Chroma store
    if persist and os.path.exists(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_function
        )
        print("Using existing vector store.")
        # Optionally, verify it has data
        if vectorstore._collection.count() > 0:
            return vectorstore
        else:
            print(f"Vector store at {persist_directory} is empty. Populating it now.")

    # If no vector store exists or it's empty, create a new one
    print(f"Creating new vector store at {persist_directory}")
    vectorstore = Chroma.from_documents(
        documents=docs_split,
        embedding=embedding_function,
        persist_directory=persist_directory if persist else None,
    )

    # Persist the vector store if specified
    if persist:
        vectorstore.persist()
        print(f"Vector store saved to {persist_directory}")

    return vectorstore


def create_vectorstore_retriever(vectorstore):
    """Create a retriever from the vector store."""

    retriever = vectorstore.as_retriever()
    print("Retriever created successfully!")

    return retriever


def retriever_tool(retriever, description="", name=""):
    """Create a retriever tool with the specified prompt."""
    print("Creating retriever tool...")
    retriever_tool = create_retriever_tool(
        retriever,
        description=description,
        name=name,
    )
    print("Retriever tool created successfully!")

    return retriever_tool
