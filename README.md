# F-rsta-ai-projekt
Bygga en ai
# -*- coding: utf-8 -*-
"""
MoodLens: En enkel känslodetektor för text
Final Project for Building AI Course
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords

# Ladda nödvändiga NLTK-data (kör första gången)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MoodLens:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        self.model = MultinomialNB()
        self.is_trained = False
        self.mood_map = {0: "😠 Arg", 1: "😔 Ledsen", 2: "😐 Neutral", 3: "😃 Glad"}
    
    def preprocess_text(self, text):
        """Förbearbetar texten genom att rensa bort onödiga tecken och göra om till gemener"""
        # Ta bort URL:er
        text = re.sub(r'http\S+', '', text)
        # Ta bort användarnämn (@username)
        text = re.sub(r'@\w+', '', text)
        # Ta bort alla icke-bokstavstecken (förutom blanksteg)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Gör om till gemener
        text = text.lower()
        # Ta bort extra blanksteg
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_and_prepare_data(self, file_path):
        """Laddar och förbearbetar träningsdata"""
        # Denna funktion förutsätter att du har en CSV-fil med 'text' och 'sentiment' kolumner
        # För detta exempel skapar vi lite exempeldata
        data = {
            'text': [
                "I am so happy today this is amazing",
                "I feel really great about this wonderful news",
                "This is fantastic I love it",
                "I am so sad and disappointed right now",
                "This is terrible I feel awful",
                "Why does everything always go wrong",
                "I am so angry about this situation",
                "This makes me furious and upset",
                "The weather is okay today",
                "I went to the store and bought some groceries",
                "This is a normal day nothing special",
                "The meeting was scheduled for tomorrow"
            ],
            'sentiment': [3, 3, 3, 1, 1, 1, 0, 0, 2, 2, 2, 2]  # 0=arg, 1=ledsen, 2=neutral, 3=glad
        }
        
        df = pd.DataFrame(data)
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        return df
    
    def train(self, data=None):
        """Tränar modellen på träningsdata"""
        if data is None:
            data = self.load_and_prepare_data("")
        
        X = data['cleaned_text']
        y = data['sentiment']
        
        # Dela upp i träning och test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Vektorisera texten
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Träna modellen
        self.model.fit(X_train_vectors, y_train)
        
        # Utvärdera modellen
        train_preds = self.model.predict(X_train_vectors)
        test_preds = self.model.predict(X_test_vectors)
        
        print(f"Träningsnoggrannhet: {accuracy_score(y_train, train_preds):.2f}")
        print(f"Testnoggrannhet: {accuracy_score(y_test, test_preds):.2f}")
        print("\nDetaljerad utvärdering:")
        print(classification_report(y_test, test_preds, 
                                   target_names=['Arg', 'Ledsen', 'Neutral', 'Glad']))
        
        self.is_trained = True
        return self
    
    def predict_mood(self, text):
        """Förutsäger känslan i given text"""
        if not self.is_trained:
            print("Varning: Modellen är inte tränad. Tränar med exempeldata...")
            self.train()
        
        # Förbearbeta texten
        cleaned_text = self.preprocess_text(text)
        
        # Vektorisera
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Gör förutsägelse
        prediction = self.model.predict(text_vector)[0]
        
        return self.mood_map.get(prediction, "Osäker")
    
    def interactive_demo(self):
        """Interaktiv demo för att testa modellen"""
        print("=== MoodLens Känslodetektor ===")
        print("Skriv en text så ska jag gissa känslan bakom den!")
        print("Skriv 'avsluta' för att sluta.")
        print("=" * 40)
        
        while True:
            user_input = input("\nDin text: ")
            if user_input.lower() == 'avsluta':
                print("Hej då!")
                break
            
            mood = self.predict_mood(user_input)
            print(f"📝 Jag uppfattar känslan som: {mood}")

# Huvudkod
if __name__ == "__main__":
    # Skapa och träna modellen
    mood_detector = MoodLens()
    mood_detector.train()
    
    # Testa med några exempel
    test_texts = [
        "I am so happy today",
        "This makes me really angry",
        "I feel sad about what happened",
        "The weather is normal today"
    ]
    
    print("🤖 Testar MoodLens med exempeltexter:")
    for text in test_texts:
        mood = mood_detector.predict_mood(text)
        print(f"'{text}' -> {mood}")
    
    # Starta interaktiv demo
    print("\n")
    mood_detector.interactive_demo()
