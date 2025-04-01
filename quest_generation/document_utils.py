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
