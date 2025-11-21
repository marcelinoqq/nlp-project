# Sentiment Analysis Project (Lightweight Version)

This version of the project is built using a **subdirectory of the dataset (100,000 rows)** instead of the full dataset.  
Because of that, **accuracy on very short reviews may not be perfect**, as the model is not trained on all available data.

---

## 📥 Dataset Download

The dataset download step is already included in **`code.ipynb`**.  
To download the full dataset:

1. Open `code.ipynb`.
2. **Uncomment the first cell**.
3. Run it — the notebook will print **exactly where KaggleHub stores the dataset** (KaggleHub does *not* save to the working directory by default).

---

## 🤖 Pretrained Models Included

The repository contains:

- `best_model.pkl`
- `tfidf_vectorizer.pkl`

You may use these directly with `app.py` **without re-training** the model.

If you prefer to retrain from scratch, simply run all cells in `code.ipynb`.

---

## 🚀 Running the Streamlit App

To launch the sentiment analysis app:

```bash
streamlit run app.py