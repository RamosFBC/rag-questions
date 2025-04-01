import sys

sys.path.append("/Users/feliperamos/Documents/eme2/true-minimalvp")
from quest_generation.vectorstore_utils import (
    create_vectorstore_retriever,
    retriever_tool,
)
from quest_generation.document_utils import load_documents, split_text
from quest_generation.env_utils import load_env
from quest_generation.ai_agent import create_graph, ToolConfig
import pprint

from langchain_core.messages import HumanMessage

load_env()

paths = "/Users/feliperamos/Documents/eme2/true-minimalvp/docs/ehae178.pdf"


# Initialize ToolConfig
tool_config = ToolConfig(paths)

# Create the graph
graph = create_graph(tool_config)

# Define initial state with tools
initial_state = {
    "messages": [HumanMessage(content="What is the treatment for hypertension?")],
    "clinical_scenario": "",
    "tools": tool_config.get_tools(),  # Pass tools into the initial state
}

# Invoke the graph
result = graph.invoke(initial_state)
print(result["messages"][-1].content)
