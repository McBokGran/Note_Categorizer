import streamlit as st
from pineconesetup import initialize_pinecone
from pineconeembedding import generate_embedding
from transformers import pipeline

# Load pipelines
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

summarizer = load_summarizer()
classifier = load_classifier()

# Initialize Pinecone
index = initialize_pinecone()

# Predefined categories
categories = ["To-Do", "Purpose of the Meeting", "What Has Been Done"]

# Function to categorize notes
def categorize_notes(notes, categories):
    categorized = {category: [] for category in categories}
    for note in notes.split("\n"):
        if note.strip():  # Skip empty lines
            results = classifier(note, candidate_labels=categories)
            top_category = results["labels"][0]
            categorized[top_category].append(note)
    return categorized

# Function to upsert data into Pinecone
def upsert_to_pinecone(categorized_notes):
    for category, notes in categorized_notes.items():
        for i, note in enumerate(notes):
            embedding = generate_embedding(note)
            metadata = {"text": note, "category": category}
            index.upsert([(f"{category}-{i}", embedding, metadata)])

# Streamlit app
st.title("Meeting Notes with Pinecone")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Paste meeting notes below or upload a text file.
2. Click "Summarize, Categorize, and Store" to process the notes.
3. Search for notes using natural language queries.
""")

# Input Section
st.header("Upload or Enter Meeting Notes")
uploaded_file = st.file_uploader("Upload a text file (optional)", type=["txt"])
if uploaded_file:
    meeting_notes = uploaded_file.read().decode("utf-8")
else:
    meeting_notes = st.text_area("Or, type your notes here:", height=300)

# Summarization, Categorization, and Upsertion
if st.button("Summarize, Categorize, and Store"):
    if meeting_notes:
        # Summarize notes
        st.header("Summary")
        summary = summarizer(meeting_notes, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        st.write(summary)

        # Categorize notes
        st.header("Categorized Notes")
        categorized_notes = categorize_notes(meeting_notes, categories)
        for category, notes in categorized_notes.items():
            st.subheader(category)
            if notes:
                for note in notes:
                    st.write(f"- {note}")
            else:
                st.write("No notes found in this category.")

        # Store in Pinecone
        upsert_to_pinecone(categorized_notes)
        st.success("Notes have been stored in Pinecone!")
    else:
        st.error("Please provide meeting notes.")

# Search Section
st.header("Search Stored Notes")
query = st.text_input("Enter your search query:")
if st.button("Search"):
    if query.strip():
        query_embedding = generate_embedding(query)
        results = index.query(query_embedding, top_k=5, include_metadata=True)

        st.subheader("Search Results")
        if results["matches"]:
            for match in results["matches"]:
                st.write(f"**Category:** {match['metadata']['category']}")
                st.write(f"**Text:** {match['metadata']['text']}")
                st.write(f"**Score:** {match['score']:.2f}")
        else:
            st.write("No relevant notes found.")
    else:
        st.error("Please enter a valid query.")
