from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from pinecone import Pinecone
import os
from uuid import uuid4


def create_vectorstore(docs_split, index_name="medical-documents"):
    """
    Create or load a persistent Chroma vector store.

    Args:
        docs_split: List of split documents to store.
        persist: Boolean to indicate if the vector store should be persisted.
        persist_directory: Directory where the vector store is saved/loaded.

    Returns:
        Chroma: The loaded or newly created vector store.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

    # Verify the index exists
    if index_name not in pc.list_indexes().names():
        raise ValueError(
            f"Index '{index_name}' does not exist. Please create it in Pinecone "
            "with dimension=1536 and metric='cosine' before running this code."
        )

    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    default_namespace_vectors = stats["namespaces"].get("", {}).get("vector_count", 0)

    # Define the embedding function
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    uuids = [str(uuid4()) for _ in range(len(docs_split))]
    if default_namespace_vectors == 0:
        print(f"Populating Pinecone index '{index_name}' with documents.")
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_function)
        vectorstore.add_documents(documents=docs_split, ids=uuids)
    else:
        print(f"Loading existing Pinecone index '{index_name}'.")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name, embedding=embedding_function
        )

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
