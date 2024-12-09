import os
from dotenv import load_dotenv
from pinecone import Pinecone, Index

# Load environment variables
load_dotenv(dotenv_path="nlpenv.env")

# Pinecone configuration
API_KEY = os.getenv("PINECONE_API_KEY")
ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")

def initialize_pinecone():
    """
    Initialize Pinecone and return the Index instance.
    """
    if not API_KEY or not ENVIRONMENT or not INDEX_NAME:
        raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, or INDEX_NAME is missing")

    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=API_KEY)

        # Check if the index exists
        if INDEX_NAME not in pc.list_indexes().names():
            raise ValueError(f"Index '{INDEX_NAME}' does not exist in environment '{ENVIRONMENT}'.")

        # Retrieve host for the index
        index_info = pc.describe_index(INDEX_NAME)
        host = index_info.host

        # Connect to the index
        index = Index(name=INDEX_NAME, host=host, api_key=API_KEY)
        print(f"Connected to Pinecone index '{INDEX_NAME}'.")
        return index
    except Exception as e:
        print("Error initializing Pinecone:", e)
        raise
