import pandas as pd
import spacy
from transformers import pipeline
import stanza

nlp_ru = spacy.load("ru_core_news_sm")
hf_ner = pipeline("ner", model="dslim/bert-base-NER")
nlp_stanza = stanza.Pipeline(lang='ru', processors='tokenize,ner')

df = pd.read_csv("./data/data.csv")
reviews = df["Review"].fillna("")

results = []
for review in reviews:
    if not isinstance(review, str):
        review = str(review)

    doc = nlp_ru(review)
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]

    hf_entities = hf_ner(review)

    doc_stanza = nlp_stanza(review)
    stanza_entities = [(ent.text, ent.type) for ent in doc_stanza.ents]

    results.append({
        "Review": review,
        "SpaCy": spacy_entities,
        "Hugging Face": hf_entities,
        "Stanza": stanza_entities
    })

results_df = pd.DataFrame(results)

results_df.to_csv("ner_results.csv", index=False)
print("Обработка завершена. Результаты сохранены в ner_results.csv.")
