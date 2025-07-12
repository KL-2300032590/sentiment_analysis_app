 Sentiment Analysis Web App

#Live demo - []


## fileInfo
# Sentiment_analysis_app
- app.py
- model.py
- requirements.txt
- sentiment_model.pkl


## model.py

A mini dataset precisely an hardcored one with 10 labeled sentences and using pipelines combines vectorizer and classifer into one model. and by using MultinomiaNB() - navie bayes is good for text classification and coming to the saves and trained model for streamlit to use (joblib.dump())

## app.py

Loads pre-trained model data using joblib.load() and then user need give an input any text aka st.text_area() and Classifies the sentiment using model.predict() it will display the result with color styling {st.success/error/info}
