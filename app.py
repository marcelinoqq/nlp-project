import streamlit as st
from predict import predict_sentiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
def set_page_config():
    """Sets up the initial page configuration."""
    st.set_page_config(
        page_title="Amazon Review Sentiment Analyzer",
        page_icon="🤖",
        layout="wide",   
        initial_sidebar_state="collapsed"
    )

# --- CUSTOM CSS & STYLING ---
def load_custom_css():
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 2rem;
                padding-right: 3rem;
                padding-left: 3rem;
                padding-bottom: 3rem;
            }
            .result-card {
                padding: 25px;
                border-radius: 12px;
                margin-top: 20px;
                color: #ffffff; 
                text-align: center;
                font-size: 28px;
                font-weight: 700;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); 
                transition: transform 0.3s ease-in-out;
            }
            .result-card:hover { transform: translateY(-3px); }
            .pos-card { background: linear-gradient(135deg, #16a085, #2ecc71); }
            .neg-card { background: linear-gradient(135deg, #e74c3c, #c0392b); }
            textarea {
                border-radius: 8px !important;
                border: 1px solid #d3d3d3 !important;
                padding: 10px !important;
            }
            .batch-item-pos {
                border-left: 6px solid #2ecc71; 
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                background-color: #f0fff4; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            .batch-item-neg {
                border-left: 6px solid #e74c3c; 
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                background-color: #fff0f0; 
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            .batch-sentiment { font-weight: 700; font-size: 1.1em; margin-bottom: 5px; color: #2c3e50; }
            .batch-item-text { font-size: 0.95em; color: #333333; }
            .batch-confidence { font-size: 0.9em; color: #555; }
            [data-testid="stMetric"] { border: none; padding: 0; }
        </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_prediction(reviews, true_labels=None):
    """
    Returns predictions and probabilities.
    If true_labels are provided, also computes evaluation metrics.
    """
    try:
        labels, probs = predict_sentiment(reviews)
        metrics_dict = None

        if true_labels is not None:
            y_true = [1 if t == "positive" else 0 for t in true_labels]
            y_pred = [1 if l == "positive" else 0 for l in labels]

            metrics_dict = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, zero_division=0)
            }

        return labels, probs, metrics_dict

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        return None, None, None

def display_single_result(label, prob):
    """Renders the result for a single review analysis."""
    if label == "positive":
        confidence = prob
        card_class = "pos-card"
        sentiment_text = "Positive Review"
        progress_val = float(prob)
    else:
        confidence = 1 - prob
        card_class = "neg-card"
        sentiment_text = "Negative Review"
        progress_val = float(1 - prob)

    st.markdown(f"<div class='result-card {card_class}'>{sentiment_text}</div>", unsafe_allow_html=True)
    st.markdown("### **Model Confidence**")
    st.progress(progress_val)
    
    # Display confidence metric
    confidence_str = f"{confidence:.2%}"
    col_metric, _ = st.columns([1, 4])
    with col_metric:
        st.metric(label="Confidence Score", value=confidence_str,
                  help=f"The model is {confidence_str} confident that the review is {label}.")

# --- SINGLE REVIEW ANALYSIS ---
# --- SINGLE REVIEW ANALYSIS ---
def analyze_single_review():
    st.header("1. Analyze a Single Review")
    st.markdown("---") 

    if "single_review_text" not in st.session_state:
        st.session_state.single_review_text = ""
    if "single_true_label" not in st.session_state:
        st.session_state.single_true_label = ""
    if "single_result" not in st.session_state:
        st.session_state.single_result = None

    review_text = st.text_area(
        "Enter your Amazon review below:",
        height=180,
        value=st.session_state.single_review_text,
        key="single_review_text",
        placeholder="Example: 'This product is amazing! Fast shipping and excellent quality.'"
    )

    true_label = st.selectbox(
        "Optional: Select the true sentiment label for this review (to see metrics)",
        ["", "positive", "negative"],
        index=0,
        key="single_true_label"
    )

    col1, _, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("🚀 Analyze Sentiment", key="single_analyze"):
            st.session_state.single_result = get_prediction(
                [review_text],
                true_labels=[true_label] if true_label else None
            )

    # Display result if available
    if st.session_state.single_result:
        labels, probs, metrics = st.session_state.single_result
        display_single_result(labels[0], probs[0])
        if metrics:
            st.markdown("### 📊 Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['Precision']:.2%}")
            col3.metric("Recall", f"{metrics['Recall']:.2%}")
            col4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")

# --- BATCH REVIEW ANALYSIS ---
def analyze_batch_reviews():
    st.header("2. Analyze Multiple Reviews (Batch Mode)")
    st.markdown("---")

    if "batch_reviews_text" not in st.session_state:
        st.session_state.batch_reviews_text = ""
    if "batch_labels_text" not in st.session_state:
        st.session_state.batch_labels_text = ""
    if "batch_result" not in st.session_state:
        st.session_state.batch_result = None

    batch_input = st.text_area(
        "Enter multiple reviews. Separate each review with an empty line.",
        height=300,
        value=st.session_state.batch_reviews_text,
        key="batch_reviews_text",
        placeholder="Review 1 (Great purchase!)\n\nReview 2 (Disappointed with the quality)..."
    )

    batch_labels_input = st.text_area(
        "Optional: Enter the true sentiment labels (positive/negative) for each review, separated by empty lines.",
        height=150,
        value=st.session_state.batch_labels_text,
        key="batch_labels_text",
        placeholder="positive\n\nnegative"
    )

    col1, _, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("📊 Run Batch Analysis", key="batch_analyze"):
            reviews = [r.strip() for r in batch_input.split("\n\n") if r.strip()]
            true_labels = [l.strip().lower() for l in batch_labels_input.split("\n\n") if l.strip()]
            if true_labels and len(true_labels) != len(reviews):
                st.warning("⚠️ Number of labels must match the number of reviews. Metrics will be skipped.")
                true_labels = None
            if reviews:
                st.session_state.batch_result = get_prediction(reviews, true_labels)

    # Display batch results if available
    if st.session_state.batch_result:
        labels, probs, metrics = st.session_state.batch_result
        reviews = [r.strip() for r in batch_input.split("\n\n") if r.strip()]
        st.subheader(f"Results for {len(reviews)} Reviews")

        # Display metrics
        if metrics:
            st.markdown("### 📊 Batch Evaluation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['Precision']:.2%}")
            col3.metric("Recall", f"{metrics['Recall']:.2%}")
            col4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")

        for text, sentiment, prob in zip(reviews, labels, probs):
            confidence = prob if sentiment == "positive" else (1 - prob)
            card_class = "batch-item-pos" if sentiment == "positive" else "batch-item-neg"
            sentiment_emoji = "✅" if sentiment == "positive" else "❌"
            html_content = f"""
                <div class='{card_class}'>
                    <p class='batch-sentiment'>{sentiment_emoji} **{sentiment.upper()}**</p>
                    <p class='batch-item-text'>{text}</p>
                    <p class='batch-confidence'>Confidence: **{confidence:.2f}**</p>
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    set_page_config()
    load_custom_css()

    center_spacer, main_col, _ = st.columns([1, 4, 1], gap="large")

    with main_col:
        st.title("Amazon Review Sentiment Analyzer 📈")
        st.markdown(
            """
            An AI-powered tool for classifying Amazon reviews as **Positive** or **Negative**.
            Built using a **TF-IDF Vectorizer** and a **Machine Learning Model**.
            """
        )
        
        st.markdown("##")
        analyze_single_review()
        
        st.markdown("##")
        analyze_batch_reviews()

        st.markdown("---")
        st.caption("USJ - ESIB")
