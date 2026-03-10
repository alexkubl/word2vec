import re
import html
import ftfy
import numpy as np
from collections import Counter

from word2vec.model import Word2Vec

def process_dataset(dataset):
    return dataset.map(
        filter_samples,
        batched=True
    )


@staticmethod
def load_model(path):

    data = np.load(path, allow_pickle=True)

    vocab_size = int(data["vocab_size"])
    embedding_dim = int(data["embedding_dim"])

    model = Word2Vec(vocab_size, embedding_dim)

    model.context_embeddings = data["context_embeddings"]
    model.target_embeddings = data["target_embeddings"]

    model.word2idx = data["word2idx"].item()
    model.idx2word = data["idx2word"].item()

    return model

def clean_text(text):
    text = html.unescape(text)
    text = text.strip()
    text = ftfy.fix_text(text)
    text = re.sub(r'\s*@-@\s*', '-', text)
    text = re.sub(r'\s*@\.@\s*', '.', text)
    text = re.sub(r'\s*@,@\s*', ',', text)
    text = re.sub(r'\s*-\s*', ' ', text)  
    text = re.sub(r'\s*—\s*', ' ', text) 
    text = re.sub(r'\s*–\s*', ' ', text)  
    text = re.sub(r'\s*\|\s*', ' ', text) 
    text = re.sub(r'-{3,}', '. ', text)
    text = re.sub(r'—{2,}', '. ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\"', "'", text)
    text = re.sub(r"\'", "'", text)
    text = re.sub(r'--', "", text)
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'\\t', '\t', text)
    text = re.sub(r'\\r', '\r', text)
    text = re.sub(r'[''`]', "'", text)
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def filter_samples(batch):
    filtered_texts = []

    for text in batch['text']:
        if re.match(r'^[=\s]+$', text) or re.match(r'^\s*=[\s=]*[^=].*=[\s=]*$', text):
            continue
        cleaned = clean_text(text)
        if cleaned:
            filtered_texts.append(cleaned)

    return {'text': filtered_texts}

def tokenize_dataset(dataset):
    corpus = []

    for text in dataset['text']:
        # replace punctuation with space, remove numbers
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()

        # split into sentences by period
        sentences = text.split(".")

        for s in sentences:
            tokens = [t for t in s.split() if len(t) > 1]
            if len(tokens) > 1:
                corpus.append(tokens)

    return corpus