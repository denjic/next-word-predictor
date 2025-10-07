import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.preprocessing import word_tokenize, load_vocab, encode


class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


# Load vocabulary
word2idx, idx2word = load_vocab('utils/vocabulary.pkl')
vocab_size = len(word2idx)
sequence_length = 4

# Load model
model = PredictiveKeyboard(vocab_size)
model.load_state_dict(torch.load('next_word_model.pth',
                      map_location=torch.device('cpu')))
model.eval()


def suggest_next_words(model, text_prompt, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())
    if len(tokens) < sequence_length - 1:
        raise ValueError(
            f"Input should be at least {sequence_length - 1} words long.")

    input_seq = tokens[-(sequence_length - 1):]
    input_tensor = torch.tensor(encode(input_seq, word2idx)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()

    return [idx2word[idx] for idx in top_indices]


# Streamlit UI
st.title("Next Word Predictor")
st.write("Enter a phrase to predict the next word (e.g., 'So, are we really').")
user_input = st.text_input("Your phrase:", "")
if user_input:
    try:
        predictions = suggest_next_words(model, user_input)
        st.write("**Top 3 predicted words:**")
        for i, word in enumerate(predictions, 1):
            st.write(f"{i}. {word}")
    except ValueError as e:
        st.error(str(e))
