import argparse
import re
from gensim.models import Word2Vec
from tqdm import tqdm
import os
import pickle

class PreprocessedCorpus:
    """
    Stream and preprocess a large text file line by line for Word2Vec training.
    """
    def __init__(self, file_path, max_length=24, min_length=3, debug=False):
        self.file_path = file_path
        self.max_length = max_length
        self.min_length = min_length
        self.valid_characters = re.compile(r"^[a-z0-9_-]+$")
        self.debug = debug
        self.total_lines = sum(1 for _ in open(file_path, 'r'))

    def __iter__(self):
        with open(self.file_path, "r") as f:
            pbar = tqdm(f, total=self.total_lines, desc="Processing tokens", 
                       unit="lines", dynamic_ncols=True)
            for line in pbar:
                token = line.strip().lower()
                if self._is_valid_token(token):
                    if self.debug:
                        pbar.write(f"Token: {token}")
                    yield [token]
                pbar.update(1)

    def _is_valid_token(self, token):
        return (
            self.min_length <= len(token) <= self.max_length and
            self.valid_characters.match(token) is not None
        )

class MinimalWordPredictor:
    """
    Minimal class that only stores word frequencies and implements prefix matching
    """
    def __init__(self):
        self.word_frequencies = {}

    @classmethod
    def from_word2vec(cls, word2vec_model):
        predictor = cls()
        # Extract only the word frequencies from the model
        for word in word2vec_model.wv.index_to_key:
            predictor.word_frequencies[word] = word2vec_model.wv.get_vecattr(word, "count")
        return predictor

    def predict(self, prefix, top_n):
        """Get top-n completions for a prefix"""
        prefix = prefix.lower()
        matches = [(word, freq) for word, freq in self.word_frequencies.items() 
                  if word.startswith(prefix)]
        
        if not matches:
            return []

        # Sort by frequency and take top-n
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:top_n]
        
        # Normalize scores
        max_freq = matches[0][1]
        return [(word, freq/max_freq) for word, freq in matches]

def train_predictor(file_path, min_count, debug):
    """
    Train Word2Vec and extract only the frequency information we need
    """
    print(f"Starting training on file: {file_path}")
    print(f"Min count: {min_count}")

    # Create and process corpus
    corpus = PreprocessedCorpus(file_path, max_length=24, min_length=3, debug=debug)

    # Train Word2Vec temporarily
    print("Building vocabulary and counting frequencies...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,  # This won't matter as we'll discard the vectors
        window=1,
        sg=1,
        min_count=min_count,
        workers=4,
        epochs=1  # One epoch is enough since we only need frequencies
    )

    # Create predictor
    print("Creating predictor...")
    predictor = MinimalWordPredictor.from_word2vec(model)

    # Save minimal model
    model_path = f"{file_path.rsplit('.', 1)[0]}.pred"
    with open(model_path, 'wb') as f:
        pickle.dump(predictor, f)

    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary size: {len(predictor.word_frequencies):,} words")
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

def retrieve_candidates(model_path, prefix, top_n):
    """
    Load model and retrieve predictions
    """
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        predictor = pickle.load(f)

    return predictor.predict(prefix, top_n)

def main():
    """
    Main function to parse CLI arguments and perform training or testing.
    """
    parser = argparse.ArgumentParser(description="Train or test a word prediction model.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test the model.")
    parser.add_argument("file", type=str, help="Path to the input text file (for training) or model file (for testing).")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum count for a token to be included in the vocabulary (default: 2) [train only].")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print tokens during training [train only].")
    parser.add_argument("--prefix", type=str, help="Prefix to predict full words for [test only].")
    parser.add_argument("--n", type=int, help="Number of top predictions to retrieve [test only].")

    args = parser.parse_args()

    if args.mode == "train":
        train_predictor(args.file, args.min_count, args.debug)
    elif args.mode == "test":
        if not args.prefix or not args.n:
            print("Error: --prefix and --n are required for test mode.")
            return
        predictions = retrieve_candidates(args.file, args.prefix, args.n)
        if predictions:
            print(f"Top {args.n} predictions for prefix '{args.prefix}':")
            for word, score in predictions:
                print(f"{word}: {score:.4f}")
        else:
            print("No predictions found.")

if __name__ == "__main__":
    main()