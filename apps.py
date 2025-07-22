import streamlit as st
import pickle

# Load model & vectorizer
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Config App ===
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #6c5ce7;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #a29bfe;'>Deteksi apakah sebuah berita palsu atau nyata berdasarkan isi teks.</p>",
    unsafe_allow_html=True
)

# === Mode Input ===
mode = st.radio("Pilih metode input:", ["📝 Input Teks Manual", "📚 Pilih dari Contoh Berita"])

# === Contoh berita ===
sample_news = {
    "Biden signs new climate bill": "President Biden signed a new climate bill into law today...",
    "Aliens discovered in Indonesia": "Aliens were found walking around in Jakarta last night...",
    "NASA announces Mars mission": "NASA confirmed next manned Mars mission in 2026..."
}

text_input = ""

if mode == "📝 Input Teks Manual":
    text_input = st.text_area("Masukkan teks berita:", height=180)
elif mode == "📚 Pilih dari Contoh Berita":
    title = st.selectbox("Pilih judul berita:", list(sample_news.keys()))
    text_input = sample_news[title]
    st.info(text_input)

# === Prediksi ===
if st.button("🔍 Prediksi"):
    if text_input.strip():
        vectorized = vectorizer.transform([text_input])
        pred = model.predict(vectorized)[0]

        # Background warna sesuai hasil
        if pred.lower() == "fake":
            bg_color = "#fab1a0"  # pink pastel
            text_color = "#d63031"
        else:
            bg_color = "#d0f0c0"  # mint
            text_color = "#00b894"

        st.markdown(
            f"<div style='background-color: {bg_color}; padding: 20px; border-radius: 10px;'>"
            f"<h4 style='color: {text_color};'>Hasil Prediksi: {pred.upper()}</h4></div>",
            unsafe_allow_html=True
        )

        # Metadata ditampilkan setelah prediksi
        st.markdown("#### 🧾 Informasi Tambahan")
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("🌍 Asal Negara:", ["USA", "UK", "Indonesia", "India", "Other"], key="country_post")
        with col2:
            category = st.selectbox("📚 Kategori Berita:", ["Politics", "Health", "Tech", "Entertainment", "Other"], key="category_post")

        # Tambahkan ringkasan metadata
        st.markdown(
            f"<p style='margin-top:10px;'>📌 <b>Negara:</b> <code>{country}</code> &nbsp;&nbsp; | &nbsp;&nbsp;"
            f"<b>Kategori:</b> <code>{category}</code></p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Teks berita tidak boleh kosong!")

# === Footer ===
st.markdown("<hr><small style='color:gray;'>Built with ❤️ using Streamlit</small>", unsafe_allow_html=True)
