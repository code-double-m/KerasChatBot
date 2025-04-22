# Simple LSTM Chatbot

This is a simple terminal-based chatbot built using TensorFlow and Keras. It uses an LSTM model to learn from a custom dataset of question-answer pairs stored in a text file.

## Files

- chatbot.py - The main script that handles data loading, preprocessing, model training, saving, loading, and running the chatbot.
- data.txt - A plain text file containing tab-separated question and answer pairs, one per line.
- chatbot_model.h5 - The saved trained model.

## Dataset Format

Each line in data.txt should contain a question and an answer separated by a tab character (\t). Example:

Hello	Hi there!  
How are you?	I'm doing well, thanks!

## Requirements

To install the necessary packages, run:

pip install tensorflow tqdm numpy

## How to Run

Make sure you have your data.txt file in the same directory. Then, run the script:

python chatbot.py

## What the Script Does

- Loads the conversation data from data.txt
- Tokenizes and pads the sequences
- Builds a sequential LSTM model with an embedding layer
- Trains the model using sparse categorical crossentropy loss and early stopping
- Saves the model as chatbot_model.h5
- Loads the model and starts a chatbot loop in the terminal

## Chatbot Usage

Once the model is trained and loaded, the chatbot will start in the terminal:

Chatbot: Hello! Type 'exit' to end the conversation.  
You: Hi  
Chatbot: how are you

To end the conversation, type 'exit'.

## Customization

You can modify the embedding dimension, the number of LSTM units, or the number of training epochs in the chatbot.py script. You can also replace data.txt with your own set of question-answer pairs to train the chatbot on a different dataset.

## Saving and Loading

The model is saved automatically as chatbot_model.h5 after training. It is then loaded before entering the chatbot loop.

