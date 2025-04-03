from flask import Flask, request, jsonify
from quest_generation.ai_agent import create_graph, ToolConfig
from langchain_core.messages import HumanMessage
import json

app = Flask(__name__)

paths = "/Users/feliperamos/Documents/eme2/true-minimalvp/docs/ehae178.pdf"


# Initialize ToolConfig
tool_config = ToolConfig(paths)


# Example endpoint: Query ChromaDB
@app.route("/query", methods=["POST"])
def query():
    data = request.json  # Expecting {"prompt": "user input"}
    # Add your ChromaDB query logic here
    # For example: results = client.some_query_method(data["prompt"])
    results = {"example": "This is a placeholder response"}  # Replace with actual logic
    return jsonify({"results": results})


@app.route("/generate-question", methods=["POST"])
def generate_question():
    data = request.json
    graph = create_graph(tool_config)
    initial_state = {
        "messages": [HumanMessage(content=data["prompt"])],
        "clinical_scenario": "",
        "tools": tool_config.get_tools(),  # Pass tools into the initial state
    }
    output = graph.invoke(initial_state)

    response = json.loads(output["messages"][-1].content)

    return jsonify({"results": response})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
