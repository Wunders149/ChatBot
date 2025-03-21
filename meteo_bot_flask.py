import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import requests
from flask import Flask, request, render_template_string
from datetime import datetime, timedelta

app = Flask(__name__)

# Fonctions météo (inchangées)
def get_current_weather(city="Mahajanga", api_key="1cdb4ab25b9bcd43335ae4cb1055b816"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"Il fait {temp} degrés avec {desc} à {city}."
    return "Désolé, je n'ai pas pu récupérer la météo."

def get_weekly_weather(city="Mahajanga", api_key="1cdb4ab25b9bcd43335ae4cb1055b816"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&lang=fr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast = []
        for i in range(0, len(data['list']), 8):
            day_data = data['list'][i]
            date = datetime.fromtimestamp(day_data['dt']).strftime('%A')
            temp = day_data['main']['temp']
            desc = day_data['weather'][0]['description']
            forecast.append(f"{date}: {temp}°C, {desc}")
        return "Prévisions pour Mahajanga :\n" + "\n".join(forecast[:5])
    return "Désolé, je n'ai pas pu récupérer les prévisions."

# Préparation des données et modèle LSTM (inchangé)
conversations = [
    ("Quel temps fait-il aujourd'hui ?", "<start> Il fait soleil aujourd'hui. <end>"),
    ("Va-t-il pleuvoir demain ?", "<start> Oui, il risque de pleuvoir demain. <end>"),
    ("Quelle est la température ?", "<start> La température est de 20 degrés. <end>"),
    ("Est-ce qu'il fait froid ?", "<start> Non, il ne fait pas très froid. <end>"),
    ("Donne-moi la météo de la semaine.", "<start> Cette semaine, attends-toi à du soleil et un peu de pluie. <end>")
]

input_texts, target_texts = zip(*conversations)
tokenizer = Tokenizer(filters='')
all_texts = list(input_texts) + list(target_texts)
tokenizer.fit_on_texts(all_texts)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(max(len(seq.split()) for seq in input_texts), max(len(seq.split()) for seq in target_texts))

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_length, padding='post')

decoder_target_data = np.zeros((len(target_sequences), max_length, vocab_size))
for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq):
        if t < len(seq) - 1:
            decoder_target_data[i, t, word_id] = 1.0

embedding_dim = 64
lstm_units = 100

encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_length,))
dec_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=100, verbose=1)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

index_to_word = {index: word for word, index in tokenizer.word_index.items()}

def generate_response(input_text):
    if "aujourd'hui" in input_text.lower() or ("temps" in input_text.lower() and "semaine" not in input_text.lower()):
        return get_current_weather()
    if "température" in input_text.lower():
        return get_current_weather()
    if "semaine" in input_text.lower():
        return get_weekly_weather()

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
    words = response.split()
    cleaned_response = []
    last_word = None
    for word in words:
        if word != last_word:
            cleaned_response.append(word)
            last_word = word
    return ' '.join(cleaned_response)

# Template HTML (inchangé)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>MétéoBot - Mahajanga</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #74ebd5, #acb6e5); margin: 0; padding: 20px; }
        .container { background: white; border-radius: 15px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); padding: 20px; max-width: 600px; margin: 0 auto; }
        h1 { text-align: center; color: #333; }
        .chat-box { border: 1px solid #ddd; border-radius: 10px; padding: 15px; height: 400px; overflow-y: auto; background: #f9f9f9; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; max-width: 80%; word-wrap: break-word; }
        .user { background: #007bff; color: white; align-self: flex-end; margin-left: auto; }
        .bot { background: #28a745; color: white; align-self: flex-start; }
        .input-area { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; outline: none; }
        input[type="text"]:focus { border-color: #007bff; }
        button, input[type="submit"] { padding: 10px 20px; border: none; border-radius: 5px; background: #007bff; color: white; cursor: pointer; }
        button:hover, input[type="submit"]:hover { background: #0056b3; }
        .reset-btn { background: #dc3545; }
        .reset-btn:hover { background: #b02a37; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MétéoBot - Mahajanga</h1>
        <div class="chat-box" id="chat">
            {% for msg in chat_history %}
                <div class="message {{ msg.type }}">{{ msg.text }}</div>
            {% endfor %}
        </div>
        <form method="POST" action="/" class="input-area">
            <input type="text" name="message" placeholder="Posez une question..." autocomplete="off">
            <input type="submit" value="Envoyer">
            <button type="submit" formaction="/reset" class="reset-btn">Réinitialiser</button>
        </form>
    </div>
    <script>
        document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
    </script>
</body>
</html>
"""

chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    global chat_history
    if request.method == 'POST':
        user_message = request.form['message']
        if user_message.strip():
            chat_history.append({"type": "user", "text": "Vous : " + user_message})
            bot_response = generate_response(user_message)
            chat_history.append({"type": "bot", "text": "MétéoBot : " + bot_response})
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route('/reset', methods=['POST'])
def reset():
    global chat_history
    chat_history = []
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Utilise le port assigné par Render ou 5000 par défaut
    app.run(host='0.0.0.0', port=port, debug=False)