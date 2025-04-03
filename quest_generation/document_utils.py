from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(paths):
    """Load documents from specified paths and split them into chunks."""
    # Load environment variables
    # Specify the paths to the PDF files
    paths = [paths]

    docs = [PyMuPDFLoader(path).load() for path in paths]
    docs_list = [item for sublist in docs for item in sublist]
    print("Documents loaded successfully!")
    return docs_list


def split_text(docs_list, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    docs_split = text_splitter.split_documents(docs_list)
    print("Documents split successfully!")

    return docs_split


from langchain.schema import Document


def add_new_document(docs_split, vectorstore):
    """
    Add document chunks to the Pinecone vector store, preserving PyMuPDFLoader metadata.

    Args:
        docs_split (list): List of document chunks from split_text (Document objects).
        vectorstore (PineconeVectorStore): The Pinecone vector store instance.
    """
    # Prepare documents with existing metadata
    documents_to_add = []
    for chunk in docs_split:
        # Ensure chunk is a Document object with metadata
        if not isinstance(chunk, Document):
            chunk = Document(page_content=str(chunk), metadata={})

        # Use the existing metadata from PyMuPDFLoader
        # Optionally, add or modify metadata if needed
        metadata = chunk.metadata.copy()  # Preserve original metadata
        metadata["chunk_id"] = (
            f"{metadata.get('author', 'unknown')}-{len(documents_to_add)}"  # Optional: unique ID
        )

        new_doc = Document(page_content=chunk.page_content, metadata=metadata)
        documents_to_add.append(new_doc)

    # Add all documents to Pinecone in one batch
    vectorstore.add_documents(documents_to_add)
    print(
        f"Added {len(documents_to_add)} chunks to Pinecone vector store with PyMuPDFLoader metadata."
    )
