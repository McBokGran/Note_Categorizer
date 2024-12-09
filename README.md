# Note_Categorizer

### Purpose of the Project
Design a meeting notes summarization and organization tool to enable users in meeting-intensive work environments/pursuit in better organization to summarize, categorize, store, and retrieve meeting notes with high accuracy, minimal manual effort, and Contextual based search capabilities.

### Solution Design
The solution ties together language models for text processing, vector database readability and functionality for storage and retrieval, as well as an interactive interface on the web. Its functions are the following: 

- Summarization of the meeting notes to bare-bones action items and insights.
- Categorizing notes to pre-set categories for organization.
- Storing notes into Pinecone as embeddings for scalable and fast semantic search.
- Retrieve relevant notes using semantic queries through intuitive user interface.

### System Components

1. Frontend:

- Streamlit App:Allows users to upload notes, view summaries and categorized notes, and search stored notes.

2. Language Models:

- Hugging Face Pipelines:

-- Summarization: facebook/bart-large-cnn
-- Classification: facebook/bart-large-mnli


3. Embedding Generation: embaas/sentence-transformers-multilingual-e5-large (or any compatible model).
   
4. Vector Database:

- Pinecone:
  
-- Stores vectorized meeting notes along with metadata.
-- Enables semantic search based on similarity scoring.
