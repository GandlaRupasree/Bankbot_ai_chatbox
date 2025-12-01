
import json, os
# Very small rule-based manager that uses intent from spaCy textcat
from typing import Dict

RESPONSES_PATH = os.path.join(os.path.dirname(__file__), 'responses.json')

def load_responses():
    with open(RESPONSES_PATH, 'r') as f:
        return json.load(f)

class ChatManager:
    def __init__(self):
        self.responses = load_responses()
        self.context = {}

    def handle(self, intent, entities):
        # simple matching: if intent in responses, return, else fallback
        if intent in self.responses:
            return self.responses[intent]
        return self.responses.get('fallback', "Sorry, I didn't get that.")

if __name__ == '__main__':
    # demo
    cm = ChatManager()
    print(cm.handle('check_balance', {}))
    print(cm.handle('apply_loan', {}))
    print(cm.handle('unknown_intent', {}))
