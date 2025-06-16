import streamlit as st
import joblib
import sklearn

# Muat model dan vektorisasi yang telah disimpan
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Layout Streamlit
st.title("Emotion Detection from Text")
st.write("Enter text in English and the system will detect the emotions contained in the text.")

# Input teks dari pengguna
user_input = st.text_area("Write text here")

if user_input:
    # Vektorisasi input
    input_tfidf = vectorizer.transform([user_input])

    # Prediksi emosi
    emotion_pred = model.predict(input_tfidf)
    emotion = label_encoder.inverse_transform(emotion_pred)
    text_output = ""

    if (emotion[0] == 0):
        text_output = "Sadness"
    elif (emotion[0] == 1):
        text_output = "Joy"
    elif (emotion[0] == 2):
        text_output = "Love"
    elif (emotion[0] == 3):
        text_output = "Anger"
    elif (emotion[0] == 4):
        text_output = "Fear"

    # Tampilkan hasil
    st.write(f"Emotion Prediction: {text_output}")