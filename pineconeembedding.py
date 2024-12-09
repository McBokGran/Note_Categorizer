from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer from embaas
tokenizer = AutoTokenizer.from_pretrained("embaas/sentence-transformers-multilingual-e5-large")
model = AutoModel.from_pretrained("embaas/sentence-transformers-multilingual-e5-large")

def generate_embedding(text):
    """
    Generate a 1024-dimensional embedding for the given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding[0]
