import os
import dotenv


def load_env():
    """Load environment variables from .env file."""
    # Load environment variables from .env file
    dotenv.load_dotenv()
    # Set OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("Environment variables loaded successfully!")
