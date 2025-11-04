import spacy
try:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Hello, world!")
    print("spaCy OK | tokens:", [t.text for t in doc])
except Exception as e:
    print("Falhou ao carregar en_core_web_sm:", e)