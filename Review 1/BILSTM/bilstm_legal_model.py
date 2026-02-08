import json
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ============== CONFIG ==============
MAX_TEXT_LEN = 500
MAX_SUMMARY_LEN = 100
EMBEDDING_DIM = 128
HIDDEN_UNITS = 128
BATCH_SIZE = 16
EPOCHS = 20

# ============== LOAD JSON DATASET ==============
def load_json_dataset(json_path='legal_dataset.json'):
    print("Loading JSON dataset...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    summaries = []
    
    for doc in data['documents']:
        judgment_text = doc['judgment']['judgment_text']
        summary_text = doc['judgment']['summary']
        
        judgment_clean = re.sub(r'[^a-zA-Z0-9\s]', '', judgment_text.lower())
        summary_clean = re.sub(r'[^a-zA-Z0-9\s]', '', summary_text.lower())
        
        texts.append(judgment_clean)
        summaries.append('sostok ' + summary_clean + ' eostok')
    
    print(f"Loaded {len(texts)} documents")
    return texts, summaries

# ============== TOKENIZATION ==============
def prepare_tokenizers(texts, summaries):
    print("Creating tokenizers...")
    
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(texts)
    
    summary_tokenizer = Tokenizer()
    summary_tokenizer.fit_on_texts(summaries)
    
    text_seq = text_tokenizer.texts_to_sequences(texts)
    summary_seq = summary_tokenizer.texts_to_sequences(summaries)
    
    text_pad = pad_sequences(text_seq, maxlen=MAX_TEXT_LEN, padding='post')
    summary_pad = pad_sequences(summary_seq, maxlen=MAX_SUMMARY_LEN, padding='post')
    
    vocab_text = len(text_tokenizer.word_index) + 1
    vocab_summary = len(summary_tokenizer.word_index) + 1
    
    print(f"Text vocab: {vocab_text}, Summary vocab: {vocab_summary}")
    
    return text_pad, summary_pad, text_tokenizer, summary_tokenizer, vocab_text, vocab_summary

# ============== BUILD SIMPLE MODEL (FIXED) ==============
def build_model(text_vocab, summary_vocab):
    print("Building BiLSTM model...")
    
    # Encoder
    encoder_input = Input(shape=(MAX_TEXT_LEN,), name='encoder_input')
    enc_emb = Embedding(text_vocab, EMBEDDING_DIM)(encoder_input)  # Removed mask_zero
    
    encoder = Bidirectional(LSTM(HIDDEN_UNITS, return_state=True, dropout=0.3), name='encoder_bilstm')
    _, fh, fc, bh, bc = encoder(enc_emb)
    
    state_h = Concatenate()([fh, bh])
    state_c = Concatenate()([fc, bc])
    
    # Decoder
    decoder_input = Input(shape=(MAX_SUMMARY_LEN,), name='decoder_input')
    dec_emb = Embedding(summary_vocab, EMBEDDING_DIM)(decoder_input)  # Removed mask_zero
    
    decoder_lstm = LSTM(HIDDEN_UNITS*2, return_sequences=True, dropout=0.3, name='decoder_lstm')
    dec_out = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Output
    decoder_dense = Dense(summary_vocab, activation='softmax', name='output_layer')
    output = decoder_dense(dec_out)
    
    model = Model([encoder_input, decoder_input], output, name='BiLSTM_Legal_Summarizer')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    return model

# ============== TRAIN ==============
def train_model(model, X_train, y_train, X_val, y_val):
    print("Starting training...")
    
    # Prepare decoder data
    dec_input_train = y_train[:, :-1]
    dec_target_train = y_train[:, 1:]
    
    dec_input_val = y_val[:, :-1]
    dec_target_val = y_val[:, 1:]
    
    # Pad decoder inputs to MAX_SUMMARY_LEN
    dec_input_train = pad_sequences(dec_input_train, maxlen=MAX_SUMMARY_LEN, padding='post')
    dec_input_val = pad_sequences(dec_input_val, maxlen=MAX_SUMMARY_LEN, padding='post')
    dec_target_train = pad_sequences(dec_target_train, maxlen=MAX_SUMMARY_LEN, padding='post')
    dec_target_val = pad_sequences(dec_target_val, maxlen=MAX_SUMMARY_LEN, padding='post')
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('bilstm_model.h5', save_best_only=True, monitor='val_loss')
    
    # Train
    history = model.fit(
        [X_train, dec_input_train],
        np.expand_dims(dec_target_train, -1),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_val, dec_input_val], np.expand_dims(dec_target_val, -1)),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return model, history

# ============== MAIN ==============
if __name__ == "__main__":
    # Load data
    texts, summaries = load_json_dataset('legal_dataset.json')
    
    # Tokenize
    X, y, text_tok, summ_tok, text_vocab, summ_vocab = prepare_tokenizers(texts, summaries)
    
    # Split 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Build model
    model = build_model(text_vocab, summ_vocab)
    
    # Train
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save
    model.save('bilstm_model.h5')
    with open('tokenizers.pkl', 'wb') as f:
        pickle.dump({'text': text_tok, 'summary': summ_tok}, f)
    
    print("\n✓ Model saved: bilstm_model.h5")
    print("✓ Tokenizers saved: tokenizers.pkl")
    print("\nNow run: python test_and_metrics.py")
