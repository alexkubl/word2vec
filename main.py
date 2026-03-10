import word2vec.utils as utils
from word2vec.model import Word2Vec
from datasets import load_dataset
import numpy as np
import argparse



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size', type=int, default=2000, help='Maximum vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of word embeddings')
    parser.add_argument('--window_size', type=int, default=2, help='Context window size')
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    
    args = parser.parse_args()

    # Load dataset and preprocess
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    dataset = utils.process_dataset(ds['train'])
    corpus = utils.tokenize_dataset(dataset)

    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    window_size = args.window_size
    neg_samples = args.neg_samples
    epochs = args.epochs
    learning_rate = args.learning_rate

    model = Word2Vec(vocab_size, embedding_dim, neg_samples)
    freq = model.build_vocab(corpus, vocab_size)
    model.prepare_negative_sampling(freq)
    model.generate_samples(corpus, window_size)
    model.train(epochs=epochs, learning_rate=learning_rate)

    model.save_model("model.npz")

    model = utils.load_model("model.npz")

    print("Vocabulary size:", len(model.word2idx))
    print("First 20 words:", list(model.word2idx.keys())[:100])
    print(model.most_similar("hot"))


if __name__ == "__main__":
    main()