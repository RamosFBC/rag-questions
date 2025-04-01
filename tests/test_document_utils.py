import sys

sys.path.append("/Users/feliperamos/Documents/eme2/true-minimalvp")
from quest_generation.document_utils import load_documents, split_text

paths = "/Users/feliperamos/Documents/eme2/true-minimalvp/docs/ehae178.pdf"

docs_list = load_documents(paths)
docs_split = split_text(docs_list)
