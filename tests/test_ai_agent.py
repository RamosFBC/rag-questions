import pytest
from langchain_core.messages import HumanMessage
from quest_generation.ai_agent import create_graph, ToolConfig
from quest_generation.env_utils import load_env


@pytest.fixture
def tool_config():
    """Fixture to initialize ToolConfig."""
    load_env()
    document_path = "/Users/feliperamos/Documents/eme2/true-minimalvp/docs/ehae178.pdf"
    return ToolConfig(document_path)


@pytest.fixture
def initial_state(tool_config):
    """Fixture to define the initial state."""
    return {
        "messages": [HumanMessage(content="What is the treatment for hypertension?")],
        "clinical_scenario": "",
        "tools": tool_config.get_tools(),
    }


def test_create_graph(tool_config):
    """Test if the graph is created successfully."""
    graph = create_graph(tool_config)
    assert graph is not None, "Graph creation failed."


def test_graph_invocation(tool_config, initial_state):
    """Test invoking the graph with an initial state."""
    graph = create_graph(tool_config)
    result = graph.invoke(initial_state)
    assert "messages" in result, "Result does not contain 'messages'."
    assert len(result["messages"]) > 0, "No messages returned in the result."
    assert (
        "content" in result["messages"][-1]
    ), "Last message does not contain 'content'."


def test_clinical_scenario_generation(tool_config):
    """Test if the clinical scenario is generated correctly."""
    graph = create_graph(tool_config)
    state = {
        "messages": [
            HumanMessage(content="Describe a clinical scenario for diabetes.")
        ],
        "clinical_scenario": "",
        "tools": tool_config.get_tools(),
    }
    result = graph.invoke(state)
    assert "clinical_scenario" in result, "Clinical scenario not generated."
    assert len(result["clinical_scenario"]) > 0, "Clinical scenario is empty."


def test_rewrite_question(tool_config):
    """Test if the question is rewritten correctly."""
    graph = create_graph(tool_config)
    state = {
        "messages": [HumanMessage(content="What is the treatment for hypertension?")],
        "clinical_scenario": "A 45-year-old patient with hypertension and no other comorbidities.",
        "tools": tool_config.get_tools(),
    }
    result = graph.invoke(state)
    assert "messages" in result, "Result does not contain 'messages'."
    assert len(result["messages"]) > 1, "No rewritten question returned."
    assert (
        "content" in result["messages"][-1]
    ), "Rewritten question does not contain 'content'."


def test_generate_question(tool_config):
    """Test if the question generation works correctly."""
    graph = create_graph(tool_config)
    state = {
        "messages": [HumanMessage(content="What is the treatment for hypertension?")],
        "clinical_scenario": "A 45-year-old patient with hypertension and no other comorbidities.",
        "tools": tool_config.get_tools(),
    }
    result = graph.invoke(state)
    assert "messages" in result, "Result does not contain 'messages'."
    assert len(result["messages"]) > 1, "No generated question returned."
    assert (
        "content" in result["messages"][-1]
    ), "Generated question does not contain 'content'."


if __name__ == "__main__":
    pytest.main()
