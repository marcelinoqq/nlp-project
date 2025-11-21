import streamlit as st
from predict import predict_sentiment

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="⚡",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}
.pos {
    background: linear-gradient(135deg, #00b16a, #2ecc71);
}
.neg {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
}
.batch-tag-pos {
    background-color: #2ecc71;
    padding: 4px 10px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
}
.batch-tag-neg {
    background-color: #e74c3c;
    padding: 4px 10px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("Sentiment Analysis")
st.write("Built with **TF-IDF + Best ML Model** + **Streamlit** for fast inference.")

st.divider()

# ============================================================
# 📝 SINGLE REVIEW SECTION
# ============================================================
st.subheader("📝 Analyze a Single Review")

review_text = st.text_area(
    "Enter your review below 👇",
    height=150,
    placeholder="Type something like: 'I love this product, highly recommended!'"
)

if st.button("🔍 Analyze Sentiment", type="primary"):
    if review_text.strip():
        with st.spinner("Analyzing..."):
            try:
                labels, probs = predict_sentiment([review_text])
                label = labels[0]
                prob = probs[0]
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

        # ---------- CARD RESULT ----------
        if label == "positive":
            st.markdown(
                f"<div class='result-card pos'>😊 Positive Review</div>",
                unsafe_allow_html=True
            )
            st.progress(float(prob))
            st.write(f"**Confidence:** `{prob:.2f}`")
        else:
            st.markdown(
                f"<div class='result-card neg'>😞 Negative Review</div>",
                unsafe_allow_html=True
            )
            st.progress(float(1 - prob))
            st.write(f"**Confidence:** `{1 - prob:.2f}`")

st.divider()

# ============================================================
# 📦 BATCH MODE
# ============================================================
st.subheader("📦 Analyze Multiple Reviews (Batch Mode)")

batch_input = st.text_area(
    "Enter one review per line 👇",
    height=200,
    placeholder="Review 1\nReview 2\nReview 3..."
)

if st.button("🚀 Analyze Batch", type="secondary"):
    lines = [line.strip() for line in batch_input.split("\n") if line.strip()]

    if len(lines) == 0:
        st.warning("⚠️ Please enter at least one review.")
    else:
        with st.spinner("Analyzing batch..."):
            try:
                labels, probs = predict_sentiment(lines)
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

        st.write("### 📊 Batch Results")

        for text, sentiment, prob in zip(lines, labels, probs):
            tag = "batch-tag-pos" if sentiment == "positive" else "batch-tag-neg"
            emoji = "😊" if sentiment == "positive" else "😞"
            confidence = prob if sentiment == "positive" else (1 - prob)

            st.markdown(
                f"""
                <div style="padding:10px 5px;">
                    <span class="{tag}">{emoji} {sentiment.upper()}</span>
                    <span style="font-size:15px; margin-left:10px;">{text}</span><br>
                    <small>Confidence: <b>{confidence:.2f}</b></small>
                </div>
                """,
                unsafe_allow_html=True
            )
