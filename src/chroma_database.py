import chromadb

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="embeddings_collection")

def store_embedding(embedding, metadata, chunk_id):
    collection.add(
        embedding=embedding.tolist(),
        metadata=metadata
    )

def retrieve_embedding(metadata):
    return collection.get(metadata=metadata)

def query_embedding(embedding, n_results=5):
    return collection.query(embedding=embedding, n_results=n_results)
