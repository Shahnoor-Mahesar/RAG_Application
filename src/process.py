from pdf_extract import get_chunks, decode_chunk
from embedding import generate_embeddings, generate_query_embedding
from chroma_database import store_embedding, retrieve_embedding, query_embedding

def process_and_store_text(long_text):
    # Split the text into chunks
    chunks = get_chunks(long_text)

    # Process each chunk and store embeddings in ChromaDB
    for i, chunk in enumerate(chunks):
        embeddings = generate_embeddings(chunk)
        metadata = {"chunk_index": i, "text": decode_chunk(chunk)}
        store_embedding(embeddings, metadata, i)

    print("Embeddings stored successfully in ChromaDB.")

def query_text(query_text):
    # Generate embedding for the query
    query_embedding_result = generate_query_embedding(query_text)

    # Perform a nearest neighbor search
    nearest_neighbors = query_embedding(query_embedding_result, n_results=5)
    return nearest_neighbors
