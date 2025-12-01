
Bank Chatbot Project - Milestone ZIP
-----------------------------------
Structure:
- milestone1/: NLU training (spaCy) scripts
  - train_nlu.py  -> trains text classifier + ner if dataset contains entities
  - test_nlu.py
- milestone2/: Dialogue manager (rule-based)
- milestone3/: Flask web chat interface (app.py + HTML/CSS/JS)
- milestone4/: Admin panel (Flask) for managing FAQs and logs
- DATASET_INSTRUCTIONS.txt -> where to place your CSV dataset

Quick start (in VS Code):
1) Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate   # Windows - or source venv/bin/activate on mac/linux
2) Install dependencies:
   pip install spacy flask pandas
   python -m spacy download en_core_web_sm
3) Place your dataset CSV at: data/bank_chatbot_dataset.csv (or set DATASET_PATH env variable)
4) Train NLU (milestone1):
   cd milestone1
   python train_nlu.py
5) Run the web app (milestone3):
   cd ../milestone3
   python app.py
6) Admin panel:
   cd ../milestone4
   python admin_app.py

Note: If you want me to customize training examples based on your actual dataset content,
tell me and I can integrate entity examples and intent labels directly into train_nlu.py.
