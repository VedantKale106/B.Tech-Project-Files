import streamlit as st
import numpy as np
import pickle
import re
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import PyPDF2

# ============== CONFIGURATION ==============
MAX_TEXT_LEN = 400
MAX_SUMMARY_LEN = 60
MODEL_FILENAME = 'legal_summarizer.keras'
TOKENIZER_FILENAME = 'tokenizers.pkl'

# ============== PAGE SETUP ==============
st.set_page_config(
    page_title="Legal AI Assistant (Custom LSTM)",
    page_icon="⚖️",
    layout="wide"
)

# ============== 1. LOAD RESOURCES (CACHED) ==============
@st.cache_resource
def load_resources():
    """
    Loads your custom trained LSTM model and tokenizers.
    Cached so it doesn't reload on every button press.
    """
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(TOKENIZER_FILENAME):
        return None, None, None
    
    try:
        # compile=False is safer for inference
        model = load_model(MODEL_FILENAME, compile=False)
        with open(TOKENIZER_FILENAME, 'rb') as f:
            tokenizers = pickle.load(f)
        return model, tokenizers['text'], tokenizers['summary']
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# ============== 2. TEXT PROCESSING FUNCTIONS ==============
def clean_extracted_text(text):
    """
    Fixes common PDF extraction artifacts (e.g., 'Coo per' -> 'Cooper').
    """
    # Remove header/footer noise (optional, basic example)
    text = re.sub(r'Page \d+', '', text)
    
    # Fix spacing issues typical in PDFs
    text = text.replace(' .', '.').replace(' ,', ',')
    text = text.replace(' -', '-')
    
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return clean_extracted_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# ============== 3. MODEL INFERENCE (LSTM) ==============
def generate_summary(text, model, text_tok, summ_tok):
    # Preprocess text
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    seq = text_tok.texts_to_sequences([clean_text])
    padded_input = pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding='post')

    # Setup Decoder
    decoder_input = np.zeros((1, MAX_SUMMARY_LEN))
    start_token = summ_tok.word_index.get('sostok', 1)
    decoder_input[0, 0] = start_token

    decoded_words = []
    seen_words = set()

    # Generation Loop
    for i in range(1, MAX_SUMMARY_LEN):
        output = model.predict([padded_input, decoder_input], verbose=0)
        
        # Get Top 3 Candidates to reduce repetition
        top_indices = np.argsort(output[0, i-1, :])[-3:][::-1]
        
        predicted_id = top_indices[0]
        word_candidate = summ_tok.index_word.get(predicted_id, '')

        # Simple anti-repetition check
        if word_candidate in seen_words and len(word_candidate) > 3:
            predicted_id = top_indices[1]

        word = summ_tok.index_word.get(predicted_id, '')

        if predicted_id == 0 or word == 'eostok':
            break

        if word and word != 'sostok':
            decoded_words.append(word)
            seen_words.add(word)

        if i < MAX_SUMMARY_LEN:
            decoder_input[0, i] = predicted_id

    return " ".join(decoded_words)

# ============== 4. HEURISTIC Q&A ==============
def answer_question(question, document_text):
    # Robust sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document_text)
    
    q_words = re.sub(r'[^\w\s]', '', question.lower()).split()
    stop_words = {'the', 'is', 'at', 'which', 'on', 'in', 'a', 'an', 'and', 'or', 'of', 'to', 'was', 'were', 'case', 'court', 'what', 'who', 'did'}
    keywords = [w for w in q_words if w not in stop_words]

    best_score = -1
    best_sentence = "Information not found in the document."

    for sentence in sentences:
        if len(sentence.split()) < 4: continue # Skip tiny fragments
        
        s_lower = sentence.lower()
        score = 0
        
        for word in keywords:
            if word in s_lower:
                score += 1
                # Boost score for facts (numbers/sections)
                if word in ['section', 'year', 'sentence', 'amount', 'ipc', 'act']:
                    score += 2
        
        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()

    return best_sentence

# ============== 5. MAIN APPLICATION ==============
def main():
    st.title("⚖️ Legal AI Assistant")
    st.caption("Powered by Custom Bi-LSTM Network")

    # Load Model
    model, text_tok, summ_tok = load_resources()
    
    if model is None:
        st.error(f"⚠️ Missing files! Please ensure '{MODEL_FILENAME}' and '{TOKENIZER_FILENAME}' are in this folder.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload Case File")
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
        
        if st.button("Reset App"):
            st.session_state.clear()
            st.rerun()

    if uploaded_file is not None:
        # Process File Only Once
        if 'doc_text' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
            with st.spinner("Extracting and cleaning text..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.session_state.doc_text = text
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.summary = None
                else:
                    st.error("Failed to extract text.")
        
        if 'doc_text' in st.session_state:
            # Layout: Left for Text, Right for AI
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📄 Document View")
                st.text_area("Extracted Text", st.session_state.doc_text, height=500)

            with col2:
                st.subheader("📝 Summary")
                if st.session_state.summary is None:
                    if st.button("Generate Summary"):
                        with st.spinner("Running LSTM Inference..."):
                            # Run the custom model
                            summary = generate_summary(st.session_state.doc_text, model, text_tok, summ_tok)
                            st.session_state.summary = summary.capitalize()
                            st.rerun()
                else:
                    st.success(st.session_state.summary)

                st.divider()
                st.subheader("🔍 Q&A Search")
                
                with st.form(key='qa_form'):
                    user_q = st.text_input("Ask a specific question:")
                    submit = st.form_submit_button("Find Answer")
                    
                    if submit and user_q:
                        ans = answer_question(user_q, st.session_state.doc_text)
                        st.info(f"**Result:** {ans}")

    else:
        st.info("👈 Upload a legal PDF to begin.")

if __name__ == "__main__":
    main()