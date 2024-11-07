from fastapi import FastAPI, Query
import os
import uvicorn
import pandas as pd
from torch.nn.functional import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

class DummyModel:
    def predict(self, X):
        return "dummy prediction"

def load_model():
    predictor = DummyModel()
    return predictor

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_bert_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return None

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state
    
    embedding = torch.mean(last_hidden_states, dim=1).squeeze()
    return embedding

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Carregar base de dados de vagas de emprego
def load_data():
    try:
        file_path = os.path.join("dataset", "dataset_preprocessed.xlsx")
        df = pd.read_excel(file_path)
        if df['content'].isnull().all():
            raise ValueError("The content column is empty.")
    except Exception as e:
        print(f"Error loading data: {e}")
        df = pd.DataFrame(columns=["title", "content"])
    return df

df = load_data()
embedding_dim = 768  # Dimensão dos embeddings BERT
autoencoder = Autoencoder(embedding_dim)

# Carregar os embeddings ajustados
new_embeddings_matrix = torch.load('embeddings/adjusted_bert_embeddings.pt')

app = FastAPI()
app.predictor = load_model()

@app.get("/hello")
def read_hello():
    return {"message": "hello world"}

@app.get("/predict")
def predict(X: str = Query(..., description="Input text for prediction")):
    result = app.predictor.predict(X)
    return {"input_value": X, "predicted_value": result, "message": "prediction successful"}

def calculate_relevance(df, query):
    query_embedding = get_bert_embedding(query.lower())
    if query_embedding is None:
        raise ValueError("Query embedding could not be generated.")
    
    query_embedding_reduced = autoencoder.encoder(query_embedding.unsqueeze(0)).detach()
    
    cosine_similarities = cosine_similarity(query_embedding_reduced, new_embeddings_matrix).squeeze()
    
    sorted_indices = torch.argsort(cosine_similarities, descending=True)
    
    return sorted_indices, cosine_similarities

@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):
    results = []
    try:
        related_docs_indices, cosine_similarities = calculate_relevance(df, query)
        for idx in related_docs_indices[:10]:  # Top 10 documentos
            print(cosine_similarities[idx].item())
            if cosine_similarities[idx].item() > 0.2:  # Limite de similaridade
                # Garantir que os índices são inteiros ao acessar o DataFrame
                idx = int(idx)  # Converter para inteiro
                cosine_similarity_value = cosine_similarities[idx].item()  # Garantir que o valor seja float
                results.append({
                    "title": df['title'].iloc[idx],
                    "content": df['content'].iloc[idx][:500],  # Limita o conteúdo a 500 caracteres
                    "relevance": cosine_similarity_value
                })
    except Exception as e:
        return {"results": results, "error": str(e), "message": "Failed to process query"}

    return {"results": results, "message": "OK"}

def run():
    uvicorn.run("main:app", host="0.0.0.0", port=8461, reload=True)

if __name__ == "__main__":
    run()