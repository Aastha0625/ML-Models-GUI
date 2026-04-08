import re
import os
import json
import nltk
from flask import jsonify

# ── PATHS ───────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# ── NLTK ────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ── CONFIG ──────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'config.json')) as f:
    config = json.load(f)

BERT_THRESHOLD = config.get('bert_threshold', 0.5)

# ── LOAD BERT ───────────────────────────────────────
bert_model = None
bert_tokenizer = None

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    bert_path = os.path.join(MODEL_DIR, 'bert_model')

    if os.path.exists(bert_path):
        bert_tokenizer = BertTokenizer.from_pretrained(
            bert_path, local_files_only=True
        )
        bert_model = BertForSequenceClassification.from_pretrained(
            bert_path, local_files_only=True
        )
        bert_model.eval()
    else:
        print("⚠️ BERT not found")

except Exception as e:
    print(f"⚠️ BERT load failed: {e}")

# ── CLEAN TEXT ──────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ── PREDICT ─────────────────────────────────────────
def predict(request):
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        predictions = []

        # ── BERT ──
        if bert_model and bert_tokenizer:
            try:
                import torch

                inputs = bert_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]

                bert_prob = float(probs[1])
                bert_pred = 1 if bert_prob >= BERT_THRESHOLD else 0

                predictions.append({
                    'model': 'BERT',
                    'label': 'Toxic' if bert_pred else 'Non-Toxic',
                    'confidence': round(float(max(probs.tolist())) * 100),
                    'toxicity': round(bert_prob * 100),
                    'accuracy': 91.5
                })

            except Exception as e:
                print("BERT prediction failed:", e)

        # ── FINAL OUTPUT ──
        if not predictions:
            return jsonify({'error': 'Model not available'}), 500

        best_model = max(predictions, key=lambda x: x['accuracy'])

        return jsonify({
            'success': True,
            'result': best_model
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500