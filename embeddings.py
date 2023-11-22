from transformers import RobertaTokenizer, RobertaModel
import torch

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# List of sentences
def embed(sentences):
    # Tokenize and encode the sentences
    encoded_inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

    # Forward pass to get the embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract the embeddings
    embeddings = outputs.last_hidden_state

    # Stack the embeddings to create a 2D tensor
    stacked_embeddings = torch.stack([embedding.squeeze() for embedding in embeddings])

    return stacked_embeddings

