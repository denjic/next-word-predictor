# Next Word Predictor

A web app that predicts the next word in a sequence using a PyTorch LSTM model trained on Sherlock Holmes stories.

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Set up a virtual environment:
   - `python -m venv env`
   - Activate: `source env/Scripts/activate` (Windows) or `source env/bin/activate` (macOS/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`
5. Access at `http://localhost:8501`

## Deployment
Deployed on Streamlit Community Cloud: [https://next-word-predictor-hdycu4zjpqpvac8qignbsy.streamlit.app/]

## Model
- Trained on Sherlock Holmes text data in Google Colab with GPU acceleration.
- Uses an LSTM architecture with 64-dimensional embeddings and 128 hidden units.
- Predicts the top-3 next words for a given phrase.

## Files
- `app.py`: Main Streamlit app.
- `next_word_model.pth`: Trained model weights.
- `utils/preprocessing.py`: Tokenization and encoding functions.
- `utils/vocabulary.pkl`: Vocabulary mapping.
- `requirements.txt`: Dependencies.
- `env/`: Virtual environment (not tracked in Git).

## Requirements
See `requirements.txt` for dependencies.
