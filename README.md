# 🧠 Yalan Haber Tespiti - Yapay Sinir Ağı (ANN) Projesi

Bu proje, **Yapay Sinir Ağı (Artificial Neural Network - ANN)** kullanarak Türkçe haberlerin doğruluk durumunu (gerçek/yalan) sınıflandırmayı amaçlayan bir metin analiz uygulamasıdır. Projede, haber metinleri üzerinde ön işleme (lemmatizasyon, durak kelime çıkarımı vb.) uygulanmış, ardından **TF-IDF vektörleştirme** ve **Keras** ile oluşturulmuş bir sinir ağı modeli kullanılarak tahminleme gerçekleştirilmiştir.

## 📁 Proje Dosyaları

- `main.py` → Ana Python betiği (model eğitimi, test, kullanıcıdan haber alarak tahmin yapma)
- `data.json` → Türkçe gerçek ve yalan haber verilerinden oluşan JSON formatında örnek veri kümesi

## ⚙️ Kullanılan Teknolojiler

- Python 3
- [spaCy](https://spacy.io/) (Türkçe dil modeli: `tr_core_news_md`)
- [NLTK](https://www.nltk.org/) (Türkçe stopword listesi)
- [scikit-learn](https://scikit-learn.org/) (TF-IDF, train-test split)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- NumPy, JSON

## 🚀 Kurulum ve Kullanım

### 1. Gerekli kütüphaneleri yükleyin

```bash
pip install numpy spacy nltk scikit-learn tensorflow
python -m spacy download tr_core_news_md

NLTK stopword listesi için:
import nltk
nltk.download('stopwords')

### 2. Uygulamayı çalıştırın
python main.py

### 3. Kullanım
Program çalıştığında önce modeli eğitir, ardından test verisi üzerinde doğruluk oranını ekrana yazdırır. Daha sonra kullanıcıdan haber metni girmesi istenir ve girilen metnin "gerçek" mi yoksa "yalan" mı olduğu tahmin edilir.

Örnek girdi:
Bir haber girin ('q' ile çık): türkiye 2025 yılında tüm araçları elektrikliye dönüştürdü

Örnek çıktı:
Tahmin: yalan
Yalan olasılığı: %87.65
Gerçek olasılığı: %12.35

🧪 Model Özeti
Giriş verisi: TF-IDF ile vektörleştirilmiş Türkçe haber metinleri (max_features=5000)

Model mimarisi:

Dense(128) – ReLU aktivasyonu

Dense(64) – ReLU aktivasyonu

Dense(2) – Softmax (çıkış katmanı)

Eğitim parametreleri:

Loss: categorical_crossentropy

Optimizer: adam

Epochs: 10

Batch size: 32

Validation split: 0.1

Değerlendirme: Test doğruluk oranı konsola yazdırılır.

📌 Notlar
Veri kümesi sınırlı ve örnekleme amaçlıdır. Gerçek uygulamalarda daha geniş ve dengeli veri seti önerilir.

Model her çalıştırmada sıfırdan eğitilir. Gelişmiş kullanım için eğitilen model .h5 formatında kaydedilip daha sonra yüklenebilir.

Bu proje sadece Türkçe haberlerle çalışacak şekilde yapılandırılmıştır.

👩‍💻 Geliştirici
Beyza Çevrim
📍 Zonguldak Bülent Ecevit Üniversitesi
🎓 Bilgisayar Mühendisliği — 2025
