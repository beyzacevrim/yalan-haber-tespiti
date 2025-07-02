import json
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Türkçe dil modeli
try:
    nlp = spacy.load("tr_core_news_md")
except OSError:
    print("Spacy Türkçe modeli eksik. Kurmak için:\npython -m spacy download tr_core_news_md")
    exit()

# Veriyi JSON dosyasından yükle
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [1 if item['label'] == "gerçek" else 0 for item in data]
    return texts, labels

# Metin ön işleme (lemmatizasyon, küçük harfe çevirme, noktalama çıkarma)
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Ana fonksiyon
def main(file_path):
    # Veriyi yükle
    texts, labels = load_data(file_path)

    # Metin ön işleme
    processed_texts = [preprocess_text(text) for text in texts]

    # TF-IDF vektörleştirme
    stop_words_turkish = stopwords.words('turkish')
    vectorizer = TfidfVectorizer(stop_words=stop_words_turkish, max_features=5000)
    X = vectorizer.fit_transform(processed_texts).toarray()
    y = to_categorical(labels, num_classes=2)

    # Eğitim ve test verisini %80 - %20 oranında böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Yapay Sinir Ağı modeli (Keras)
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 sınıf: yalan, gerçek

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Test değerlendirme
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Başarı Oranı: %{accuracy * 100:.2f}")

    # Kullanıcıdan giriş alarak tahmin yapma
    while True:
        user_news = input("\nBir haber girin ('q' ile çık): ")
        if user_news.lower() == 'q':
            break
        processed = preprocess_text(user_news)
        vectorized = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vectorized)[0]
        predicted_label = "gerçek" if prediction[1] > prediction[0] else "yalan"
        print(f"Tahmin: {predicted_label}")
        print(f"Yalan olasılığı: %{prediction[0]*100:.2f}")
        print(f"Gerçek olasılığı: %{prediction[1]*100:.2f}")

# Dosya yolu
file_path = "data.json"

if __name__ == "__main__":
    main(file_path)
