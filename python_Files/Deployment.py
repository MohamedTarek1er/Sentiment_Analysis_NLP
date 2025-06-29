import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load(r"C:\Users\moham\Downloads\NLP_Project\svm_model.pkl")
vectorizer = joblib.load(r"C:\Users\moham\Downloads\NLP_Project\tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

class SentimentApp:
    def __init__(self):
        self.title = "🧠 Sentiment Analysis App"
        self.description = "Enter a Review or Sentence"
    
    def run(self):
        st.title(self.title)
        st.markdown(self.description)

        user_input = st.text_area("✏️ Write something:")

        if st.button("🔍 Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text.")
            else:
                cleaned_text = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_input)

                if prediction[0] == 1:
                    label = "😊 Positive"
                    st.markdown(f"<h3 style='color: green;'>Sentiment: {label}</h3>", unsafe_allow_html=True)
                else:
                    label = "😞 Negative"
                    st.markdown(f"<h3 style='color: red;'>Sentiment: {label}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    app = SentimentApp()
    app.run()
