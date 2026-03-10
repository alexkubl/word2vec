# word2vec


### Task 

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

### Overview

This repository contains the implementation of Skip-Gram Word2Vec with Negative Sampling architecture, including the following steps:

- **Vocabulary construction**: Loading ***wikitext-2*** dataset from HuggingFace; preprocessing the loaded data by cleaning the dataset artifacts; tokenizing the sentences into words.
- **Samples generation**: Creating the pairs for positive (exsisting in the initial text pairs) and negative samples – (target, context) pairs using a sliding window.
- **Training**: Model training on the resulted pairs, updating the target and context embedding matricies.

### Run the code
The requirements for running the code are specifies in `requirements.txt`.

Run model training:
```bash 
python main.py
```
with optional arguments: 
```bash 
python main.py \
    --vocab_size 1000 \
    --embedding_dim 100 \
    --window_size 2 \
    --neg_samples 4 \
    --epochs 5 \
    --learning_rate 0.01 
```

This will start the training with the provided parameters, prining the loss value for each epoch and the test result with the words similar to a word *hot* in the model vocabulary.