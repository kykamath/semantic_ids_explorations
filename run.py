import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def get_movie_embeddings():
    print("Loading MovieLens 100k dataset...")
    # Load the MovieLens 100k dataset from the official source
    # dataset = load_dataset("movielens", "100k")
    dataset = load_dataset("ashraq/movielens_ratings")

    # The 'movielens' dataset contains ratings, and movie titles are repeated.
    # We need to get the unique movie titles from the 'movie_title' feature.
    # The 'train' split contains all the data.
    movie_titles = sorted(list(set(item['movie_title'] for item in dataset['train'])))

    print(f"Loaded {len(movie_titles)} unique movie titles.")

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
