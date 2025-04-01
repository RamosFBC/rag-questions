import sys

sys.path.append("/Users/feliperamos/Documents/eme2/true-minimalvp")
from quest_generation.vectorstore_utils import (
    create_vectorstore_retriever,
    retriever_tool,
)
from quest_generation.document_utils import load_documents, split_text
from langchain.prompts import PromptTemplate
from quest_generation.env_utils import load_env

load_env()

paths = "/Users/feliperamos/Documents/eme2/true-minimalvp/docs/ehae178.pdf"

docs_list = load_documents(paths)
docs_split = split_text(docs_list)

retriever = create_vectorstore_retriever(docs_split)

retriever_prompt = "retrieve_medical_references. Search and return information necessary to make evidenced based questions. Always use this tool before generating questions."


retriever_tool = retriever_tool(retriever, retriever_prompt)
