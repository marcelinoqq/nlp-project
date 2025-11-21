import joblib
import numpy as np

model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(text_list):
    X = tfidf.transform(text_list)

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]    
    else:
        raw = model.decision_function(X)
        probs = 1 / (1 + np.exp(-raw))      

    labels = ["positive" if p == 1 else "negative" for p in preds]

    return labels, probs
