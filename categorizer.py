import streamlit as st
import pickle
import re
import string
from scipy.sparse import hstack

# Model ve vektörleştiricileri yükleme
with open('vectors/news_tr_word.pkl', 'rb') as handle:
    vectorizer_word = pickle.load(handle)

with open('vectors/news_tr_char.pkl', 'rb') as handle:
    vectorizer_char = pickle.load(handle)

with open('models/interpress_news_category_tr_lite_classifier_svm_model_4000.sav', 'rb') as handle:
    model = pickle.load(handle)

# Etiketlerin tanımlanması
labels = {
    0 : "Kültür-Sanat",
    1 : "Ekonomi",
    2 : "Siyaset",
    3 : "Eğitim",
    4 : "Dünya",
    5 : "Spor",
    6 : "Teknoloji",
    7 : "Magazin",
    8 : "Sağlık",
    9 : "Gündem"
}

# Metin temizleme fonksiyonları
def clean_text(content):
    content = content.lower()
    content = re.sub(r'\S*@\S*\s?', '', content)  # E-posta temizleme
    content = re.sub(r'http\S+', '', content)  # URL temizleme
    content = content.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini temizleme
    content = content.translate(str.maketrans('', '', string.digits))  # Sayıları temizleme
    content = ' '.join(word for word in content.split() if len(word) > 2)  # Kısa kelimeleri temizleme
    return content

# Streamlit başlığı
st.title("Metin Sınıflandırma Uygulaması")

# Kullanıcıdan metin girişi al
user_input = st.text_area("Metin girin:", height=300)

# Kullanıcıdan metin girişi alındıysa işleme başla
if st.button("Kategori Belirle"):
    if user_input:
        cleaned_input = clean_text(user_input)

        # Metni vektörleştir
        tfidf_word_vector = vectorizer_word.transform([cleaned_input])
        tfidf_char_vector = vectorizer_char.transform([cleaned_input])

        # Her iki vektörleştiriciyi birleştirme
        combined_vector = hstack([tfidf_word_vector, tfidf_char_vector])

        # Model ile tahmin yap
        category_index = model.predict(combined_vector)[0]
        category = labels[category_index]

        # Sonucu ekrana yazdır
        st.write(f"Kategori: {category}")
    else:
        st.write("Lütfen metin giriniz.")
