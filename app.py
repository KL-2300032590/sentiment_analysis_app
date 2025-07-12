import streamlit as st
import joblib

model = joblib.load('sentiment_model.pkl')

st.title("Sentiment Analysis App")
st.subheader("Enter ur text below to analyze sentiment")

user_input = st.text_area("Type your sentences here...")

if st.button("Analyze"):
    if user_input.strip()=="":
        st.warning("please enter some text!")
    else:
        prediction = model.predict([user_input])
        sentiment = prediction[0].capitalize()

        if sentiment == 'Positive':
            st.success(f"sentiment: {sentiment}")
        elif sentiment == 'Negative':
            st.error(f"Sentiment: {sentiment}")
        else:
            st.info(f"Sentiment: {sentiment}")
