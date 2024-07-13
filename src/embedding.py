from transformers import BertModel
import torch

# Initialize the BERT model
model = BertModel.from_pretrained('bert-base-uncased')

def generate_embeddings(chunk):
    input_ids = torch.tensor([chunk])
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state.squeeze(0).mean(dim=0).detach().numpy()
    
    return embeddings


def generate_query_embedding(query):
    tokens = tokenizer.encode(query, add_special_tokens=True)
    return generate_embeddings(tokens)