
import spacy, os
MODEL = os.environ.get('NLU_MODEL', 'milestone1/nlu_model')
try:
    nlp = spacy.load(MODEL)
except Exception as e:
    print('Could not load model from', MODEL, '->', e)
    exit(1)

tests = [
    "what is my account balance",
    "how do i apply for a personal loan",
    "where is the nearest atm",
]
for t in tests:
    doc = nlp(t)
    print('TEXT:', t)
    print('INTENT SCORES:', doc.cats)
    if doc.ents:
        print('ENTITIES:', [(ent.text, ent.label_) for ent in doc.ents])
    print('---')
