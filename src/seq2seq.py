from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import numpy as np







def extract_conversations(filename):

    print("Loading corpus data...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    conversations = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc="Processing", unit="line"):
            if '\t' in line:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    question, answer = parts
                    conversations.append((question, answer))
    return conversations




file_path = 'data.txt'
conversations = extract_conversations(file_path)



# Separate input and output
questions, answers = zip(*conversations)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences for fixed input length
max_sequence_length = max(max(map(len, question_sequences)), max(map(len, answer_sequences)))
question_sequences_padded = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')
answer_sequences_padded = pad_sequences(answer_sequences, maxlen=max_sequence_length, padding='post')

# Build model
embedding_dim = 50  #change this based on requirements
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(100, return_sequences=True))  # Set return_sequences=True
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()



# Train the model
epochs = 100
model.fit(question_sequences_padded, np.array(answer_sequences_padded), epochs=epochs, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

# Save the model
model.save("chatbot_model.h5")
print("Model saved as 'chatbot_model.h5'")





# Function to preprocess user input
def preprocess_input(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=max_sequence_length, padding='post')
    return user_input_padded

# Function to generate a response
def generate_response(user_input):
    user_input_padded = preprocess_input(user_input)
    response_sequence = model.predict(user_input_padded)
    response_sequence = np.argmax(response_sequence, axis=-1)
    response_text = tokenizer.sequences_to_texts(response_sequence)[0]
    return response_text




# Load the trained model
model = load_model("chatbot_model.h5")




# Chatbot loop
print("Chatbot: Hello! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    response = generate_response(user_input)
    print("Chatbot:", response)














