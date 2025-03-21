import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import requests
from datetime import datetime, timedelta

# Fonction pour la météo actuelle
def get_current_weather(city="Mahajanga", api_key="1cdb4ab25b9bcd43335ae4cb1055b816"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"Il fait {temp} degrés avec {desc} à {city}."
    return "Désolé, je n'ai pas pu récupérer la météo."

# Fonction pour la météo hebdomadaire (prévisions sur 5 jours avec l'endpoint forecast)
def get_weekly_weather(city="Mahajanga", api_key="1cdb4ab25b9bcd43335ae4cb1055b816"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&lang=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast = []
        for i in range(0, len(data['list']), 8):  # Prendre une entrée par jour (8 = 24h/3h)
            day_data = data['list'][i]
            date = datetime.fromtimestamp(day_data['dt']).strftime('%A')
            temp = day_data['main']['temp']
            desc = day_data['weather'][0]['description']
            forecast.append(f"{date}: {temp}°C, {desc}")
        return "Prévisions pour Mahajanga :\n" + "\n".join(forecast[:5])  # Limiter à 5 jours
    return "Désolé, je n'ai pas pu récupérer les prévisions."

# 1️⃣ Préparation des données avec des conversations sur la météo
conversations = [
    ("Quel temps fait-il aujourd'hui ?", "<start> Il fait soleil aujourd'hui. <end>"),
    ("Va-t-il pleuvoir demain ?", "<start> Oui, il risque de pleuvoir demain. <end>"),
    ("Quelle est la température ?", "<start> La température est de 20 degrés. <end>"),
    ("Est-ce qu'il fait froid ?", "<start> Non, il ne fait pas très froid. <end>"),
    ("Donne-moi la météo de la semaine.", "<start> Cette semaine, attends-toi à du soleil et un peu de pluie. <end>")
]

input_texts, target_texts = zip(*conversations)

tokenizer = Tokenizer(filters='')  # Garde tous les caractères
all_texts = list(input_texts) + list(target_texts)
tokenizer.fit_on_texts(all_texts)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(max(len(seq.split()) for seq in input_texts),
                 max(len(seq.split()) for seq in target_texts))

# Conversion en séquences numériques
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Padding pour uniformiser les longueurs
encoder_input_data = pad_sequences(input_sequences, maxlen=max_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_length, padding='post')

# Préparation des données cibles pour le teacher forcing
decoder_target_data = np.zeros((len(target_sequences), max_length, vocab_size))
for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq):
        if t < len(seq) - 1:  # Exclure <end> pour le teacher forcing
            decoder_target_data[i, t, word_id] = 1.0

# 2️⃣ Modèle Encodeur-Décodeur avec LSTM
embedding_dim = 64
lstm_units = 100

# Encodeur
encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Décodeur
decoder_inputs = Input(shape=(max_length,))
dec_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compilation du modèle
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 3️⃣ Entraînement du modèle
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=100, verbose=1)

# 4️⃣ Configuration pour l'inférence
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# Fonction de génération de réponse
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

def generate_response(input_text):
    # Météo actuelle
    if "aujourd'hui" in input_text.lower() or "temps" in input_text.lower() and "semaine" not in input_text.lower():
        return get_current_weather()
    # Température actuelle
    if "température" in input_text.lower():
        return get_current_weather()
    # Prévisions hebdomadaires
    if "semaine" in input_text.lower():
        return get_weekly_weather()

    # Réponse générée par LSTM pour autres cas
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_seq, maxlen=max_length, padding='post')

    states_value = encoder_model.predict(padded_input, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    decoded_sentence = []
    end_token_id = tokenizer.word_index['<end>']
    max_response_len = max_length

    while len(decoded_sentence) < max_response_len:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_to_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or sampled_word == '':
            break
        if sampled_word != '<start>':
            decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    response = ' '.join(decoded_sentence) if decoded_sentence else "Désolé, je ne peux pas répondre à ça."
    # Éviter les répétitions
    words = response.split()
    cleaned_response = []
    last_word = None
    for word in words:
        if word != last_word:
            cleaned_response.append(word)
            last_word = word
    return ' '.join(cleaned_response)

# 5️⃣ Interface graphique Tkinter
def send_message():
    user_text = user_input.get()
    if user_text.strip():
        chat_display.insert(tk.END, "Vous : " + user_text + "\n")
        bot_response = generate_response(user_text)
        chat_display.insert(tk.END, "MétéoBot : " + bot_response + "\n\n")
        user_input.delete(0, tk.END)

root = tk.Tk()
root.title("MétéoBot - Mahajanga")

chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
chat_display.pack(padx=10, pady=10)

frame_input = tk.Frame(root)
frame_input.pack(pady=5)

user_input = tk.Entry(frame_input, width=40)
user_input.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(frame_input, text="Envoyer", command=send_message)
send_button.pack(side=tk.RIGHT)

root.mainloop()