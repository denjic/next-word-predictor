import nltk
import pickle
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def encode(seq, word2idx):
    return [word2idx.get(word, word2idx.get('<unk>', 0)) for word in seq]