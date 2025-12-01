
import os
import spacy
import pandas as pd
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# Adjust DATASET_PATH if you keep dataset elsewhere
DATASET_PATH = os.environ.get("DATASET_PATH", "data/bank_chatbot_dataset.csv")
MODEL_OUTPUT = "nlu_model"

def load_data(path):
    """Expect CSV with columns: 'text' and 'intent'. Optional 'entities' column with JSON list of {'start','end','label'}"""
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'intent' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'intent' columns.")
    examples = []
    for _, row in df.iterrows():
        text = str(row['text'])
        intent = str(row['intent'])
        ents = []
        if 'entities' in df.columns and not pd.isna(row.get('entities')):
            try:
                ents = json.loads(row['entities'])
            except Exception:
                ents = []
        examples.append((text, {"cats": {intent: 1.0}, "entities": ents}))
    return examples, list(df['intent'].unique())

def train():
    examples, intents = load_data(DATASET_PATH)
    nlp = spacy.blank("en")
    # create textcat_multilabel for intent classification
    textcat = nlp.add_pipe("textcat", last=True)
    for intent in intents:
        textcat.add_label(intent)
    # optional entity recognizer
    ner = nlp.add_pipe("ner", last=False)
    # collect labels from entities if present
    for _, meta in examples:
        for ent in meta.get('entities', []):
            ner.add_label(ent['label'])

    # convert examples
    spacy_examples = []
    for text, meta in examples:
        doc = nlp.make_doc(text)
        annotations = {}
        annotations['cats'] = meta['cats']
        annotations['entities'] = [(e['start'], e['end'], e['label']) for e in meta.get('entities', [])]
        spacy_examples.append(Example.from_dict(doc, {'cats': annotations['cats'], 'entities': annotations['entities']}))

    optimizer = nlp.initialize()
    for epoch in range(5):
        random.shuffle(spacy_examples)
        losses = {}
        batches = minibatch(spacy_examples, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            nlp.update(batch, drop=0.2, sgd=optimizer, losses=losses)
        print(f"Epoch {epoch+1} Losses: {losses}")

    nlp.to_disk(MODEL_OUTPUT)
    print("Model trained and saved to", MODEL_OUTPUT)

if __name__ == '__main__':
    import json, sys
    if not os.path.exists(DATASET_PATH):
        print(f'ERROR: dataset not found at {DATASET_PATH}. Update DATASET_PATH or place dataset there.')
        sys.exit(1)
    train()
