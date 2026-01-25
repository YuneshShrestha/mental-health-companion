import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

def transform_text(text):
    """Preprocess text for prediction"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

# Page config
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Cyberbullying Detection")
st.markdown("Detect potentially harmful or bullying content in text using Machine Learning")

# Load model
try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model files not found. Please ensure model.pkl and vectorizer.pkl exist.")

if model_loaded:
    # Input
    text_input = st.text_area("Enter text to analyze:", height=150, placeholder="Type or paste text here...")

    if st.button("Analyze", type="primary"):
        if text_input.strip():
            # Preprocess and predict
            transformed = transform_text(text_input)
            vectorized = vectorizer.transform([transformed])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]

            st.markdown("---")

            if prediction == 1:
                st.error("⚠️ **Cyberbullying Detected**")
                st.metric("Confidence", f"{probability[1]*100:.1f}%")
            else:
                st.success("✅ **No Cyberbullying Detected**")
                st.metric("Confidence", f"{probability[0]*100:.1f}%")
        else:
            st.warning("Please enter some text to analyze.")

# About section
with st.expander("About this project"):
    st.markdown("""
    **Model**: Multinomial Naive Bayes with TF-IDF Vectorization

    **Dataset**: Trained on 448,000+ labeled comments from Wikipedia and social media

    **Accuracy**: ~90% on test set

    **Use Case**: Identifying potentially harmful content in online communications
    """)

st.markdown("---")
st.caption("Built with Streamlit | IIC Quest Hackathon Project")
