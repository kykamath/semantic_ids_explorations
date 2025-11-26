import torch
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def get_movie_embeddings():
    print("Loading MovieLens 100k dataset...")
    # Load the MovieLens 100k dataset
    dataset = load_dataset("mcprado/movielens_100k")

    # Extract movie titles from the 'items' split
    # The dataset structure might vary, assuming 'title' is the relevant column in 'items'
    movies_df = pd.DataFrame(dataset['items'])
    movie_titles = movies_df['title'].tolist()

    print(f"Loaded {len(movie_titles)} movie titles.")

    # Load a pre-trained sentence transformer model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for movie titles
    print("Generating movie embeddings (this might take a while)...")
    movie_embeddings = model.encode(movie_titles, show_progress_bar=True)

    print("Movie embeddings generated.")
    return movie_titles, movie_embeddings


if __name__ == '__main__':
    print("\n--- Movie Embeddings ---")
    titles, embeddings = get_movie_embeddings()
    print(f"First 5 movie titles: {titles[:5]}")
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"Embedding for the first movie: {embeddings[0][:5]}...") # print first 5 dimensions
