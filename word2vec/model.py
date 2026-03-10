from sys import path

import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, neg_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.samples_set = []
        self.neg_samples = neg_samples
        self.neg_sampling_prob = None
        self.context_embeddings = np.random.rand(vocab_size, embedding_dim)
        self.target_embeddings = np.random.uniform(
            -0.5/embedding_dim,
            0.5/embedding_dim,
            (vocab_size, embedding_dim)
        )
        self.word2idx = {}
        self.idx2word = {}

    def generate_samples(self, corpus, window_size):
        for sentence in corpus:
            sent = [w for w in sentence if w in self.word2idx]
            for i, word in enumerate(sent):
                target = self.word2idx[word]
                left = max(0, i - window_size)
                right = min(len(sent), i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    context_word = sent[j]
                    context = self.word2idx[context_word]
                    self.samples_set.append((target, context))

    def build_vocab(self, dataset, vocab_size, min_count=5, top_freq_percent=0.01):
        freq = {}
        total_words = 0
        for sentence in dataset:
            for w in sentence:
                freq[w] = freq.get(w, 0) + 1
                total_words += 1

        # Remove words below min_count
        freq = {w:c for w,c in freq.items() if c >= min_count}
        # Remove words that are too frequent
       # Sort words by frequency descending
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])

        # Remove the top `top_freq_percent` of words
        num_to_remove = int(len(sorted_words) * top_freq_percent)
        if num_to_remove > 0:
            sorted_words = sorted_words[num_to_remove:]
        # Sort and keep top max_vocab
        vocab = sorted_words[:vocab_size]

        for i, (w, _) in enumerate(vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w

        self.vocab_size = len(self.word2idx)

        return freq

    def prepare_negative_sampling(self, freq_dict):
        counts = np.array([freq_dict[self.idx2word[i]] for i in range(self.vocab_size)])
        prob = counts ** 0.75
        prob = prob / prob.sum()
        self.neg_sampling_prob = prob

    def get_negative_samples(self, positive_idx):
        negatives = np.random.choice(
            self.vocab_size,
            size=self.neg_samples,
            replace=False,
            p=self.neg_sampling_prob
        )
        negatives = [n for n in negatives if n != positive_idx]
        return negatives

    def forward(self, target_idx, context_idx):

        v_w = self.target_embeddings[target_idx]
        v_c = self.context_embeddings[context_idx]

        score = sigmoid(np.dot(v_w, v_c))

        return score

    def train(self, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            random.shuffle(self.samples_set)
            total_loss = 0
            for target, context in self.samples_set:
                v_w = self.target_embeddings[target].copy()
                v_c = self.context_embeddings[context].copy()
                # positive example
                score = sigmoid(np.dot(v_w, v_c))
                grad = score - 1
                self.context_embeddings[context] -= learning_rate * grad * v_w
                self.target_embeddings[target] -= learning_rate * grad * v_c
                total_loss += -np.log(score + 1e-10)
                # negative samples
                negatives = self.get_negative_samples(context)
                negatives = self.get_negative_samples(context)
                neg_vectors = self.context_embeddings[negatives]
                scores_neg = sigmoid(neg_vectors @ v_w)
                grad_neg = scores_neg[:, None]
                self.context_embeddings[negatives] -= learning_rate * grad_neg * v_w
                self.target_embeddings[target] -= learning_rate * np.sum(
                    grad_neg * neg_vectors,
                    axis=0
                )
                total_loss += -np.sum(np.log(1 - scores_neg + 1e-10))
            print(f"Epoch {epoch+1} Loss: {total_loss/len(self.samples_set):.4f}")
    
    def save_model(self, path="word2vec_model.npz"):
        np.savez(
            path,
            context_embeddings=self.context_embeddings,
            target_embeddings=self.target_embeddings,
            word2idx=self.word2idx,
            idx2word=self.idx2word,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        )
    
    def get_word_vector(self, word):
        if word not in self.word2idx:
            raise ValueError("Word not in vocabulary")
        idx = self.word2idx[word]
        return self.target_embeddings[idx]

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
        )

    def contains(self, word):
        return word in self.word2idx

    def most_similar(self, word, top_k=10):
        print("Finding similar words for:", word)
        if word not in self.word2idx:
            raise ValueError("Word not in vocabulary")
        word_vec = self.get_word_vector(word)
        similarities = []
        for idx, other_word in self.idx2word.items():
            if other_word == word:
                continue
            other_vec = self.target_embeddings[idx]
            sim = self.cosine_similarity(word_vec, other_vec)
            similarities.append((other_word, sim))
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
