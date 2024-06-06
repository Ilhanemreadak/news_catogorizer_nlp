import streamlit as st
import pickle
import re
import string
from scipy.sparse import hstack

with open('vectors/news_tr_word.pkl', 'rb') as handle:
    vectorizer_word = pickle.load(handle)

with open('vectors/news_tr_char.pkl', 'rb') as handle:
    vectorizer_char = pickle.load(handle)

with open('models/interpress_news_category_tr_lite_classifier_svm_model_4000.sav', 'rb') as handle:
    model = pickle.load(handle)

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

def clean_text(content):
    content = content.lower()
    content = re.sub(r'\S*@\S*\s?', '', content)  # E-posta temizleme
    content = re.sub(r'http\S+', '', content)  # URL temizleme
    content = content.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini temizleme
    content = content.translate(str.maketrans('', '', string.digits))  # Sayıları temizleme
    content = ' '.join(word for word in content.split() if len(word) > 2)  # Kısa kelimeleri temizleme
    return content

st.title("Metin Sınıflandırma Uygulaması")

user_input = st.text_area("Metin girin:", height=300)

if st.button("Kategori Belirle"):
    if user_input:
        cleaned_input = clean_text(user_input)

        tfidf_word_vector = vectorizer_word.transform([cleaned_input])
        tfidf_char_vector = vectorizer_char.transform([cleaned_input])

        combined_vector = hstack([tfidf_word_vector, tfidf_char_vector])

        category_index = model.predict(combined_vector)[0]
        category = labels[category_index]

        st.write(f"Kategori: {category}")
    else:
        st.write("Lütfen metin giriniz.")
