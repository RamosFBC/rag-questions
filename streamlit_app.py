import streamlit as st
from langchain_core.messages import HumanMessage
from quest_generation.ai_agent import create_graph, ToolConfig
import requests
import ast
import json
import dotenv
import os

dotenv.load_dotenv()
CORRECT_PASSWORD = os.getenv("APP_PASSWORD")

tool_config = ToolConfig()


def check_password():
    """Returns True if password is correct, False otherwise."""

    def password_entered():
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # Clear password
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.text_input(
            "Enter Password",
            type="password",
            key="password",
            on_change=password_entered,
        )
        return False
    return True


# Function to generate a question using the Flask API
def generate_question(prompt):
    graph = create_graph(tool_config)
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "clinical_scenario": "",
        "tools": tool_config.get_tools(),  # Pass tools into the initial state
    }
    output = graph.invoke(initial_state)

    response = json.loads(output["messages"][-1].content)

    return response


# Initialize session states:
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "alternatives" not in st.session_state:
    st.session_state["alternatives"] = []
if "alt_exp" not in st.session_state:
    st.session_state["alt_exp"] = []
if "explanation" not in st.session_state:
    st.session_state["explanation"] = ""
if "learning_objective" not in st.session_state:
    st.session_state["learning_objective"] = ""
if "edit_question" not in st.session_state:
    st.session_state["edit_question"] = False


# Title of the app
st.title("Criador de Questões")

if check_password():
    # Text input for the prompt
    user_prompt = st.text_area(
        "Escreva o que gostaria de cobrar na sua questão:",
        height=150,
        key="user_prompt",
    )

    # Button to generate response
    if st.button("Gerar Questão"):
        if user_prompt:
            response = generate_question(user_prompt)
            if response:
                # Access the JSON structure using keys
                st.session_state["question"] = response["question"]
                st.session_state["alternatives"] = response["alternatives"]
                st.session_state["alt_exp"] = response["alt_explanations"]
                st.session_state["explanation"] = response["question_explanation"]
                st.session_state["learning_objective"] = response["learning_objective"]

            else:
                st.error("Falha ao gerar a questão.")
        else:
            st.warning("Please enter a prompt first!")
    # Display the response if it exists in session state
    if st.button("Editar Questão"):
        st.session_state["edit_question"] = True
        st.rerun()
    if "question" in st.session_state and not st.session_state["edit_question"]:

        st.subheader("Questão Gerada:")
        st.write("**Enunciado:** " + st.session_state["question"])
        st.write("**Alternativas:** ")
        for i, alt in enumerate(st.session_state["alternatives"]):
            st.write(alt)
        st.write("**Explicação das Alternativas** ")
        for i, alt in enumerate(st.session_state["alt_exp"]):
            st.write(alt)
        st.write("**Explicação:** " + st.session_state["explanation"])
        st.write("**Objetivo Educacional:** " + st.session_state["learning_objective"])

    elif "question" in st.session_state and st.session_state["edit_question"]:
        # Formulario para editar a questao
        st.subheader("Edit Response:")
        with st.form("edit_form"):
            updated_question = st.text_area(
                "**Pergunta:**", value=st.session_state["question"]
            )
            updated_alternatives = st.text_area(
                "**Alternativas:**",
                value="\n".join(
                    ast.literal_eval(str(st.session_state["alternatives"]))
                ),
                height=200,
            )
            updated_alt_exp = st.text_area(
                "**Explicação das Alternativas:**",
                value="\n".join(ast.literal_eval(str(st.session_state["alt_exp"]))),
                height=300,
            )
            updated_explanation = st.text_area(
                "**Explicação:**", value=st.session_state["explanation"], height=200
            )
            updated_learning_objective = st.text_area(
                "**Objetivo Educacional:**",
                value=st.session_state["learning_objective"],
            )
            submitted = st.form_submit_button("Save Changes")

        if submitted:
            st.session_state["question"] = updated_question
            st.session_state["alternatives"] = updated_alternatives.split("\n")
            st.session_state["alt_exp"] = updated_alt_exp.split("\n")
            st.session_state["explanation"] = updated_explanation
            st.session_state["learning_objective"] = updated_learning_objective
            st.session_state["edit_question"] = False
            st.rerun()

    # Optional: Add some instructions or info
    st.sidebar.markdown(
        """
    ### Instructions
    1. Escreva o assunto que gostaria de cobrar na sua questão na área de texto
    2. Clique em "Gerar Questão" para ver a questão gerada
    """
    )
else:
    st.error("Senha incorreta. Tente novamente.")
