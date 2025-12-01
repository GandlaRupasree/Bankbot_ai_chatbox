
from flask import Flask, render_template, request, jsonify
import spacy, os

app = Flask(__name__)
# load model if exists
MODEL_DIR = os.environ.get('NLU_MODEL', 'milestone1/nlu_model')
nlp = None
try:
    nlp = spacy.load(MODEL_DIR)
except Exception:
    nlp = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/message', methods=['POST'])
def message():
    data = request.json or {}
    text = data.get('text', '')
    if nlp:
        doc = nlp(text)
        # pick top intent
        intent = max(doc.cats, key=doc.cats.get) if doc.cats else 'fallback'
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        # basic responses
        if intent == 'check_balance':
            reply = 'To check balance securely, please login via app or visit branch.'
        elif intent == 'find_atm':
            reply = 'Please share your city or use location.'
        else:
            reply = 'Sorry, I did not understand. Type "help" to see options.'
    else:
        intent, ents, reply = 'fallback', [], 'NLU model not loaded. Run milestone1/train_nlu.py first.'
    return jsonify({'intent': intent, 'entities': ents, 'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
