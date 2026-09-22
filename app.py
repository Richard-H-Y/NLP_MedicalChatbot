import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Pastikan NLTK sudah terdownload
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# --- 1. SETUP MODEL & PREPROCESSING (Di-cache agar cepat dan efisien RAM) ---
@st.cache_resource
def load_models_and_data():
    # Coba load SBERT
    try:
        from sentence_transformers import SentenceTransformer, util
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        use_sbert = True
    except Exception:
        sbert_model = None
        util = None
        use_sbert = False

    # NLTK Init
    lemmatizer = WordNetLemmatizer()
    indonesian_stopwords = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
        'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
        'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
        'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
        'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
        'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
        'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
        'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
    }
    english_stopwords = set(stopwords.words('english'))
    all_stopwords = indonesian_stopwords | english_stopwords

    # Load Data Dataset Anda (Pastikan file CSV ini sefolder dengan file app.py)
    df = pd.read_csv("re_train_embeds.csv")
    
    def parse_tags(raw):
        return [t.strip() for t in re.findall(r"'([^']*)'", str(raw)) if t.strip()]

    df = df[['short_question', 'short_answer', 'tags']].copy()
    df.columns = ['question', 'answer', 'tags_raw']
    df['tags_list'] = df['tags_raw'].apply(parse_tags)
    df['category'] = df['tags_list'].apply(lambda t: t[0].title() if t else 'Umum')
    df = df.drop(columns=['tags_raw'])
    df = df[(df['question'].str.strip() != '') & (df['answer'].str.strip() != '')].reset_index(drop=True)

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    df['processed_question'] = df['question'].apply(preprocess_text)

    # Definisi Engine Pencari (Menggantikan Class di Jupyter)
    class Engine:
        def __init__(self, dataframe):
            self.df = dataframe
            self.threshold = 0.35 if use_sbert else 0.15
            self.top_k = 3
            
            if use_sbert:
                self.sbert_embeddings = sbert_model.encode(self.df['question'].tolist(), convert_to_tensor=True)
            
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_question'])

            self.rules = {
                'emergency': {
                    'patterns': [
                        r'(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)',
                        r'(severe chest pain|can.?t breathe|cannot breathe|not breathing|unconscious|heart attack)',
                    ],
                    'responses': ["🚨 **DARURAT! Hubungi 119 atau segera ke IGD terdekat!**"]
                },
                'greeting': {
                    'patterns': [r'\b(halo|hai|hi|hello|hey)\b'],
                    'responses': ["👋 Halo! Ada keluhan yang bisa saya bantu jawab?"]
                }
            }

        def get_response(self, user_input, conversation_history):
            if not user_input.strip(): return "Silakan ketik pertanyaan Anda."
            
            # Cek Rule Darurat
            for intent, data in self.rules.items():
                for pattern in data['patterns']:
                    if re.search(pattern, user_input.lower()):
                        return random.choice(data['responses'])

            # Pakai history percakapan sebagai konteks
            query = conversation_history[-1] + " " + user_input if len(conversation_history) > 0 else user_input

            if use_sbert:
                emb = sbert_model.encode(query, convert_to_tensor=True)
                scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
                top_results = np.argsort(-scores)[:self.top_k]
                results = [(idx, float(scores[idx])) for idx in top_results]
                method = "SBERT"
            else:
                processed = preprocess_text(query)
                vec = self.vectorizer.transform([processed])
                scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
                top_results = np.argsort(scores)[::-1][:self.top_k]
                results = [(idx, scores[idx]) for idx in top_results]
                method = "TF-IDF"

            best_idx, best_score = results[0]
            if best_score < self.threshold:
                return "🤔 Saya tidak menemukan jawaban yang cukup relevan di database saya. Mohon perjelas gejalanya."

            # Keyword Boost
            boosted = []
            for idx, score in results:
                text = self.df.iloc[idx]['question']
                bonus = sum(1 for word in user_input.split() if word in text)
                boosted.append((idx, score + 0.05 * bonus))

            best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
            row = self.df.iloc[best_idx]

            # Bentuk Response menggunakan Markdown (Bawaan Streamlit)
            return f"**Kategori: {row['category']}** `({method})`\n\n{row['answer']}\n\n---\n*⚠️ Disclaimer: Ini hanya saran edukatif, bukan pengganti konsultasi medis profesional.*"

    return Engine(df), len(df)


# --- 2. MULAI STREAMLIT UI ---
# Pastikan ini dipanggil paling atas setelah import
st.set_page_config(page_title="MedBot v3", page_icon="🏥", layout="centered")

# Eksekusi fungsi setup (Akan loading SBERT saat pertama kali buka aja)
with st.spinner('Menyiapkan Model & Dataset AI...'):
    bot, df_len = load_models_and_data()

st.title("🏥 MedBot v3 — Smart Medical Assistant")
st.caption(f"🟢 Online · Menjawab dari {df_len} FAQ (re_train_embeds.csv) · Semantic Engine")

# Menyimpan riwayat chat (Memory) dan Konteks
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Halo! Saya **MedBot v3**. Coba tanyakan keluhan kesehatanmu, misal: *chest pain radiating to arm* atau *gejala diabetes*."}
    ]
if "context" not in st.session_state:
    st.session_state.context = []

# Fitur Quick Chips sebagai Tombol
st.write("**Topik Populer:**")
col1, col2, col3, col4, col5 = st.columns(5)
quick_prompts = []
if col1.button("🤕 Headache"): quick_prompts.append("severe headache")
if col2.button("🩸 Diabetes"): quick_prompts.append("diabetes sugar level")
if col3.button("🤰 Pregnancy"): quick_prompts.append("missed period pregnancy")
if col4.button("😰 Anxiety"): quick_prompts.append("anxiety stress")
if col5.button("😴 Sleep"): quick_prompts.append("trouble sleeping insomnia")

# Tombol Bersihkan Chat
if st.button("Clear Chat 🗑️", type="tertiary"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.session_state.context = []
    st.rerun()

st.divider()

# Render seluruh History Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Cek Input: dari bar bawah text box, ATAU dari tombol chip
user_input = st.chat_input("Ketik pertanyaan kesehatan...")
if len(quick_prompts) > 0:
    user_input = quick_prompts[0]

# Jika ada interaksi user
if user_input:
    # Tampilkan chat bubble si User
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Proses & tampilkan jawaban Bot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ MedBot sedang berpikir...")
        time.sleep(0.5) # Sedikit delay agar efek loading terlihat
        
        response = bot.get_response(user_input, st.session_state.context)
        message_placeholder.markdown(response)
        
    # Simpan ke riwayat state
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.context.append(user_input) # Simpan konteks untuk obrolan berikutnya