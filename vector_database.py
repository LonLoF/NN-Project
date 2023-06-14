# For faiss you have two options !pip install faiss-gpu or !pip install faiss-cpu
# The faiss-gpu package provides CUDA-enabled indices, either package should be installed, but not both.

#It is standalone file check strings_ranked_by_relatedness_vector args!
#Add this to your main file: from vector_database import save_index, load_index, strings_ranked_by_relatedness_vector
import faiss  # for vector database
import scipy.spatial.distance as spatial
import pandas as pd
import numpy as np
import time


def save_index(embeddings: list, index_path: str = 'vector_database.index') -> faiss.IndexFlatL2:
    """ Creates, saves and returns embeddings vectors"""
    # Create a vector index
    dimension = len(embeddings[0])  # Assuming all embeddings have the same dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance is used for similarity search
    
    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Add embeddings to the index
    index.add(embeddings_np)

    # Save the index
    faiss.write_index(index, index_path)
    
    return index

def load_index(index_path: str = 'vector_database.index') -> faiss.IndexFlatL2:
    """ Loads index file from disc and returns it as faiss.IndexFlatL2"""
    return faiss.read_index(index_path)
    
# Search function using the vector index and DataFrame
def strings_ranked_by_relatedness_vector(
    query: str,
    index: faiss.IndexFlatL2,
    df: pd.DataFrame,
    openai_api,
    model: str = 'text-embedding-ada-002',
    relatedness_fn=lambda x, y: 1 - spatial.cosine(x, y),
    top_n: int = 5,
    timeit: bool = False
) -> tuple[list[str], list[float]]:
    """Returns a list of top_n strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai_api.Embedding.create(
        model=model,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]

    # Start the timer
    start_time = time.time()

    # Perform similarity search using the vector database
    _, indices = index.search(np.array([query_embedding]), top_n)
    
    strings = df.loc[indices[0], "text"].tolist()
    embeddings = df.loc[indices[0], "embedding"].tolist()
    relatednesses = [relatedness_fn(query_embedding, emb) for emb in embeddings]

    # End the timer
    end_time = time.time()

    if timeit: 
        print(f'Elapsed time: {end_time - start_time} seconds')

    return strings, relatednesses