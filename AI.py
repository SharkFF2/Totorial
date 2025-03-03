import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

MINE_KLASSER = {0: 'annet', 1: 'sport', 2: 'politikk', 3: 'teknologi'}
classes = np.array([0, 1, 2, 3])

MODEL_FILENAME = 'incremental_text_classifier.pkl'
VECTORIZER_FILENAME = 'vectorizer.pkl'

mapping = {
    'rec.sport.baseball': 1,
    'rec.sport.hockey': 1,
    'talk.politics.guns': 2,
    'talk.politics.mideast': 2,
    'talk.politics.misc': 2,
    'sci.electronics': 3,
    'sci.med': 3,
    'sci.space': 3,
}

logging.info("Laster datasettet...")
newsgroups = fetch_20newsgroups(subset='all')
y_tilpasset = np.array([mapping.get(newsgroups.target_names[i], 0) for i in newsgroups.target])

if os.path.exists(VECTORIZER_FILENAME):
    logging.info("Laster eksisterende vectorizer...")
    with open(VECTORIZER_FILENAME, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    logging.info("Oppretter ny vectorizer og trener den...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(newsgroups.data)
    with open(VECTORIZER_FILENAME, 'wb') as f:
        pickle.dump(vectorizer, f)

X = vectorizer.transform(newsgroups.data)

if os.path.exists(MODEL_FILENAME):
    logging.info("Laster eksisterende modell...")
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
else:
    logging.info("Oppretter ny SGD-modell og trener den...")
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    model.partial_fit(X, y_tilpasset, classes=classes)

def hent_ny_data(mappe_sti):
    ny_data = []
    ny_etiketter = []
    for filnavn in os.listdir(mappe_sti):
        if filnavn.endswith(('.txt', '.md', '.log')):
            filbane = os.path.join(mappe_sti, filnavn)
            try:
                with open(filbane, 'r', encoding='utf-8', errors='ignore') as f:
                    innhold = f.read()
                    ny_data.append(innhold)
                    if 'sport' in filnavn.lower():
                        ny_etiketter.append(1)
                    elif 'politikk' in filnavn.lower():
                        ny_etiketter.append(2)
                    elif 'teknologi' in filnavn.lower():
                        ny_etiketter.append(3)
                    else:
                        ny_etiketter.append(0)
            except Exception as e:
                logging.warning(f"Kunne ikke lese {filbane}: {e}")
    return ny_data, ny_etiketter

mappe_sti = r'C:\Users\Felix Falldalen\OneDrive - Innlandet fylkeskommune\AI'

nye_data, nye_etiketter = hent_ny_data(mappe_sti)

if len(nye_data) > 0:
    X_ny = vectorizer.transform(nye_data)
    model.partial_fit(X_ny, nye_etiketter, classes=classes)

    logging.info(f"Oppdatert med {len(nye_data)} nye dokumenter.")
    y_pred_nye = model.predict(X_ny)
    for i, pred in enumerate(y_pred_nye):
        logging.info(f"Fil {i + 1}: Forventet {MINE_KLASSER[nye_etiketter[i]]}, Predikert {MINE_KLASSER[pred]}")

    accuracy_nye = accuracy_score(nye_etiketter, y_pred_nye)
    logging.info(f"Nøyaktighet på egne data: {accuracy_nye}")
    report_nye = classification_report(nye_etiketter, y_pred_nye, target_names=[MINE_KLASSER[i] for i in classes])
    logging.info(report_nye)

    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)

    logging.info("Modellen er oppdatert og lagret.")
else:
    logging.info("Ingen oppdatering nødvendig.")
