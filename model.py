import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

#sample dataset

data = {
    'text': [
        'I love this product',
        'This is the worst experience ever',
        'I am happy with the service',
        'Totally disappointed and frustrated',
        'Not bad, could be better',
        'Absolutely fantastic!',
        'I hate this',
        'Very good, I am satisfied',
        'Terrible, will not buy again',
        'It is okay, not great'
    ],
    'sentiment': [
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral'
    ]
}


##dataframe
df = pd.DataFrame(data)

x_trian ,x_test,y_train,y_test = train_test_split(df['text'],df['sentiment'],test_size=0.2,random_state=42)

# build pipeline

model = Pipeline([
    ('vectorizer',CountVectorizer()),
    ('classifier',MultinomialNB())
])

## train model
model.fit(x_trian,y_train)

##save model
joblib.dump(model,'sentiment_model.pkl')

print("Model trained and saved as 'sentiment_model.pkl' ")
