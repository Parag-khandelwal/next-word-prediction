
---

# Next Word Prediction using LSTM

This project implements a next word prediction model using LSTM (Long Short-Term Memory) neural networks. The model is trained on a text dataset and can predict the next word given a sequence of words.

## Prerequisites

- Python 3
- TensorFlow
- Keras
- NumPy

## Getting Started

1. Clone this repository to your local machine or open a Jupyter Notebook environment.
2. Mount Google Drive to access the text dataset file (`next_word_pred.txt`):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run the provided code cells in the notebook (`next_word_prediction.ipynb`).

## Code Description

1. **Data Preparation:**
   - Read the text file containing training data.
   - Tokenize the text using the Tokenizer from Keras.
   - Create input sequences and their corresponding labels.

2. **Model Building:**
   - Define a Sequential model in Keras.
   - Add layers for Embedding, LSTM, and Dense (softmax activation for multi-class classification).
   - Compile the model with categorical cross-entropy loss and Adam optimizer.

3. **Model Training:**
   - Train the model using the prepared input sequences and labels.
   - Monitor validation loss and accuracy during training.

4. **Model Evaluation:**
   - Save the trained model (`NWP.h5`) for future use.
   - Use the trained model to predict the next word given a starting text.


