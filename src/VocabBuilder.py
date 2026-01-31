import json
from collections import Counter

class VocabBuilder:
    def __init__(self):
        # Define special tokens
        # 0 is usually reserved for padding and 1 for unknown tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    # Builds the vocabulary from a pandas Series of tokenized text
    def build(self, tokenized_series):
        word_counts = Counter()
        for tokens in tokenized_series:
            word_counts.update(tokens)

        most_common_words = sorted(word_counts, key=word_counts.get, reverse=True)

        # Add unique words to dictionaries word2idx and idx2word starting from indices and words
        for i,word in enumerate(most_common_words):
            self.word2idx[word]=i+2 
            self.idx2word[i+2]=word

        print(f"Vocab built. Size: {len(self.word2idx)}")
    
    # Converts a list of tokens into a list of indices.
    def transform(self, tokens):
        return [self.word2idx.get(token, 1) for token in tokens]

    # Save the dictionary index as JSON to a file
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f)




