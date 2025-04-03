from typing import Annotated, Sequence, TypedDict, Literal, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from .env_utils import load_env
from .document_utils import load_documents, split_text
from .vectorstore_utils import (
    create_vectorstore_retriever,
    create_retriever_tool,
    create_vectorstore,
)

import json
import os


class ToolConfig:
    """Class to manage and initialize tools for the workflow."""

    def __init__(
        self, document_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        """Initialize tools with configurable parameters."""
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tools: List[BaseTool] = self._initialize_tools()

    def _initialize_tools(self) -> List[BaseTool]:
        """Private method to initialize the retriever tool."""
        # Load environment variables
        load_env()
        # Load and split documents
        docs_list = load_documents(self.document_path)
        docs_split = split_text(
            docs_list, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        # Create vectorstore with Pinecone
        vectorstore = create_vectorstore(docs_split, index_name="medical-documents")
        # Create retriever and tool
        retriever = create_vectorstore_retriever(vectorstore)
        retriever_prompt = "retrieve_medical_references. Search and return information necessary to make evidence-based questions. Always use this tool before generating questions."

        retriever_tool = create_retriever_tool(
            retriever, description=retriever_prompt, name="retriever_tool"
        )
        return [retriever_tool]

    def get_tools(self) -> List[BaseTool]:
        """Public method to access initialized tools."""
        return self.tools


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    clinical_scenario: str
    tools: List[BaseTool]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a grader assessing the relevance of a retrieved document to a user request in a medical context. Here is the retrieved document: {context} Here is the user request: {question} If the document contains keywords or semantic meaning related to the medical topic of the user question, grade it as relevant. Provide a binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    clinical_scenario = state["clinical_scenario"]

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    tools = state["tools"]
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def create_clinical_scenario(state):
    """
    Create a clinical case scenario based on the question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the clinical case scenario
    """
    print("---CREATE CLINICAL CASE---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
            Based on the following user question, generate a detailed clinical case scenario in Portuguese that sets up a situation where a medical student would need to apply specific medical knowledge to make a diagnosis, choose a treatment, or understand a medical concept. The scenario should include a patient’s history, symptoms, diagnostic tests, and relevant medical context to create a realistic and educational case. Ensure that the scenario naturally leads to a question or decision point that can be used to assess the student's understanding. Here is the user question: {question}""",
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    return {"clinical_scenario": response.content}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    clinical_scenario = state["clinical_scenario"]

    msg = [
        HumanMessage(
            content=f""" \n 
    Given the original user question and the generated clinical scenario, reformulate the question in Portuguese to better target the retrieval of medical documents that provide information relevant to the medical decision or concept highlighted in the clinical scenario. The improved question should combine key elements from both the original question and the clinical scenario to ensure the retrieved documents are highly relevant. Original question: {question} Clinical scenario: {clinical_scenario}""",
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    class generate_question(BaseModel):
        """Structured Output for question generation"""

        enunciate: str = Field(
            description="Enunciate of the question to be generated. \n"
            "The enunciate must be a question that assesses the student's understanding of the content\n"
            "asket by the user request. NAO DIGA 'BASEADO NO DOCUMENTO FORNECIDO' APENAS ESCREVA O ENUNCIADO. Just ask the question.\n"
        )
        alternatives: list[str] = Field(
            description="A list of 5 items, each containing one alternative of the generated question.\n"
            "Only one alternative can be right"
        )
        alt_explanations: list[str] = Field(
            description="A list of 5 items containing the explanation for each one of the alternatives\n"
            "The explanations indices must match the corresponding alternatives indices"
        )
        question_explanation: str = Field(
            description="An overall explanation of the content of the question that was generated\n"
            "so the student that answered the question may deepen its understanding and \n"
            "revisit the content asked"
        )
        learning_objective: str = Field(
            description="A short expression or phrase that briefly summarizes the content asked by the \n"
            "generated question"
        )

    docs = last_message.content

    clinical_scenario = state["clinical_scenario"]

    # Prompt
    prompt = PromptTemplate(
        template="""
Você é um professor criando uma questão de múltipla escolha para uma avaliação de conhecimento médico. Você receberá um cenário clínico, uma solicitação do usuário e um documento recuperado com informações médicas relevantes.

Sua tarefa é formular uma questão em português que avalie o entendimento do aluno sobre os conceitos médicos necessários para abordar a situação descrita no cenário clínico, com base nas informações do documento recuperado.

Aqui está o cenário clínico: {clinical_scenario}
Aqui está a solicitação do usuário: {question}
Aqui está o documento recuperado: {context}

Instruções:
1. Escreva a questão em português. 
2. O enunciado da questão deve incluir explicitamente uma quantidade considerável de detalhes relevantes do cenário clínico, como idade do paciente, sintomas, histórico médico ou resultados de exames, para fornecer contexto suficiente para que o aluno responda sem precisar consultar o cenário separadamente.
3. Certifique-se de que a questão esteja diretamente relacionada ao conteúdo médico necessário para entender ou resolver o cenário clínico, conforme informado pela solicitação do usuário e pelo documento.
4. Forneça cinco alternativas (A a E), com apenas uma resposta correta.
5. Para cada alternativa, forneça explicação em português, justificando por que ela está correta ou incorreta. A explicação deve conter os conceitos médicos envolvidos na alternativa e o motivo aprofundado da justificação. Se ela for correta deve conter início: (CORRETA), caso seja incorreta deve conter início (INCORRETA)
6. Forneça uma explicação geral da questão em português, detalhando os conceitos médicos envolvidos.
7. Indique o objetivo de aprendizagem em português, resumindo o principal conhecimento médico que a questão está testando.

Garanta que a questão seja clara, concisa e eficaz para testar o entendimento do aluno sobre os conceitos médicos relevantes.
""",
        input_variables=["context", "question", "clinical_scenario"],
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    llm_with_tool = llm.with_structured_output(generate_question)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm_with_tool

    # Run
    response = rag_chain.invoke(
        {"context": docs, "question": question, "clinical_scenario": clinical_scenario}
    )
    response = response.dict()
    response_message = {
        "role": "assistant",
        "content": json.dumps(
            {
                "question": response["enunciate"],
                "alternatives": response["alternatives"],
                "alt_explanations": response["alt_explanations"],
                "question_explanation": response["question_explanation"],
                "learning_objective": response["learning_objective"],
            },
            ensure_ascii=False,
        ),
    }
    return {"messages": [response_message]}


def create_graph(tool_config: ToolConfig):
    """
    Create a state graph for the agent.
    Args:
        tools (list): List of tools to be used in the graph.
    Returns:
        graph: The compiled state graph.
    """
    print("---CREATE GRAPH---")
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    workflow.add_node(
        "create_clinical_scenario", create_clinical_scenario
    )  # create clinical scenario
    tools = tool_config.get_tools()
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "create_clinical_scenario")
    workflow.add_edge("create_clinical_scenario", "agent")

    # Decide whether to retrieve
    workflow.add_edge("agent", "retrieve")

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()
    print("---GRAPH CREATED---")

    return graph
