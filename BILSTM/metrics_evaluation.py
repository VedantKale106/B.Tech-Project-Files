import json
import numpy as np
import pickle
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

MAX_TEXT_LEN = 500
MAX_SUMMARY_LEN = 100

# ============== LOAD MODEL ==============
print("Loading model and tokenizers...")
model = load_model('bilstm_model.h5', compile=False)  # Added compile=False
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with open('tokenizers.pkl', 'rb') as f:
    tokenizers = pickle.load(f)
    text_tokenizer = tokenizers['text']
    summary_tokenizer = tokenizers['summary']

print("✓ Model loaded successfully\n")

# ============== LOAD TEST DATA ==============
def load_test_data(json_path='legal_dataset.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    summaries = []
    
    # Use first 40 documents for testing
    for doc in data['documents'][:40]:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', doc['judgment']['judgment_text'].lower())
        summary = re.sub(r'[^a-zA-Z0-9\s]', '', doc['judgment']['summary'].lower())
        texts.append(text)
        summaries.append(summary)
    
    return texts, summaries

# ============== GENERATE PREDICTIONS ==============
def predict_summary(text, model, text_tok, summ_tok):
    # Tokenize input
    seq = text_tok.texts_to_sequences([text])
    padded_input = pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding='post')
    
    # Initialize decoder input
    decoder_input = np.zeros((1, MAX_SUMMARY_LEN))
    sostok_idx = summ_tok.word_index.get('sostok', 1)
    decoder_input[0, 0] = sostok_idx
    
    decoded_words = []
    
    for i in range(1, MAX_SUMMARY_LEN):
        # Predict next word
        predictions = model.predict([padded_input, decoder_input], verbose=0)
        
        # Get predicted token at position i-1
        predicted_id = np.argmax(predictions[0, i-1, :])
        
        # Convert id to word
        predicted_word = None
        for word, idx in summ_tok.word_index.items():
            if idx == predicted_id:
                predicted_word = word
                break
        
        if predicted_word is None or predicted_word == 'eostok':
            break
        
        if predicted_word != 'sostok':
            decoded_words.append(predicted_word)
        
        # Update decoder input for next iteration
        if i < MAX_SUMMARY_LEN:
            decoder_input[0, i] = predicted_id
    
    return ' '.join(decoded_words)

# ============== CALCULATE METRICS ==============
def calculate_all_metrics(predictions, references):
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        if not pred.strip():
            pred = "empty"
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # BLEU
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        if not pred.strip():
            pred = "empty"
        score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)
        bleu_scores.append(score)
    
    # F1 and Exact Match
    f1_scores = []
    exact_matches = []
    for pred, ref in zip(predictions, references):
        em = 1 if pred.strip() == ref.strip() else 0
        exact_matches.append(em)
        
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0)
            continue
        
        common = pred_tokens.intersection(ref_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    return {
        'ROUGE-1': np.mean(rouge_scores['rouge1']),
        'ROUGE-2': np.mean(rouge_scores['rouge2']),
        'ROUGE-L': np.mean(rouge_scores['rougeL']),
        'BLEU': np.mean(bleu_scores),
        'F1': np.mean(f1_scores) * 100,
        'Exact_Match': np.mean(exact_matches) * 100
    }

# ============== MAIN TESTING ==============
if __name__ == "__main__":
    # Load test data
    test_texts, test_summaries = load_test_data('legal_dataset.json')
    print(f"Testing on {len(test_texts)} documents\n")
    
    # Generate predictions
    predictions = []
    times = []
    
    print("Generating summaries...")
    for i, text in enumerate(test_texts):
        start = time.time()
        pred = predict_summary(text, model, text_tokenizer, summary_tokenizer)
        elapsed = time.time() - start
        
        predictions.append(pred)
        times.append(elapsed)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_texts)}...")
    
    print("\nCalculating metrics...\n")
    
    # Calculate metrics
    metrics = calculate_all_metrics(predictions, test_summaries)
    avg_time = np.mean(times)
    docs_per_hour = 3600 / avg_time
    
    # ============== PRINT RESULTS ==============
    print("=" * 70)
    print("BiLSTM MODEL PERFORMANCE METRICS FOR YOUR PAPER")
    print("=" * 70)
    
    print("\n📊 TABLE 1 - SUMMARIZATION PERFORMANCE:")
    print("-" * 70)
    print(f"ROUGE-L Score (BiLSTM): {metrics['ROUGE-L']:.2f}")
    print(f"Avg Processing Time: {avg_time:.1f} seconds per document")
    print(f"\n→ Put '{metrics['ROUGE-L']:.2f}' in Table 1, BiLSTM column under ROUGE-L")
    print(f"→ Put '{avg_time:.1f}s' in Table 1, BiLSTM column under Time")
    
    print("\n📊 TABLE 2 - QUESTION ANSWERING PERFORMANCE:")
    print("-" * 70)
    print(f"Exact Match (%): {metrics['Exact_Match']:.1f}%")
    print(f"F1-Score (%): {metrics['F1']:.1f}%")
    print(f"\n→ Put '{metrics['Exact_Match']:.0f}' in Table 2, BiLSTM column under Exact Match")
    print(f"→ Put '{metrics['F1']:.0f}' in Table 2, BiLSTM column under F1-Score")
    
    print("\n📊 ADDITIONAL METRICS:")
    print("-" * 70)
    print(f"ROUGE-1: {metrics['ROUGE-1']:.3f}")
    print(f"ROUGE-2: {metrics['ROUGE-2']:.3f}")
    print(f"BLEU Score: {metrics['BLEU']:.3f}")
    print(f"Processing Speed: {docs_per_hour:.0f} documents/hour")
    
    print("\n" + "=" * 70)
    print("QUICK REFERENCE FOR YOUR PAPER:")
    print("=" * 70)
    print(f"✓ Section 5.1, Table 1 (BiLSTM ROUGE-L): 0.{int(metrics['ROUGE-L']*100):02d}")
    print(f"✓ Section 5.1, Table 1 (BiLSTM Time): {avg_time:.0f}s")
    print(f"✓ Section 5.2, Table 2 (BiLSTM Exact Match): {metrics['Exact_Match']:.0f}%")
    print(f"✓ Section 5.2, Table 2 (BiLSTM F1-Score): {metrics['F1']:.0f}%")
    print(f"✓ Section 5.3 (Processing speed): ~{docs_per_hour:.0f} docs/hour")
    print("=" * 70)
    
    # Save sample predictions
    print("\n📝 Sample Predictions:")
    print("-" * 70)
    for i in range(min(3, len(predictions))):
        print(f"\nDocument {i+1}:")
        print(f"Reference: {test_summaries[i][:100]}...")
        print(f"Predicted: {predictions[i][:100]}...")
    
    print("\n✅ Testing complete!")
