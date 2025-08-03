import os
import tempfile
import streamlit as st
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from st_audiorec import st_audiorec

# Create directories
os.makedirs("corpus", exist_ok=True)
os.makedirs("audio", exist_ok=True)

st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

# Helper: Save audio file
def save_audio_file(audio_bytes, dialect_label):
    filename = f"{dialect_label}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.wav"
    filepath = os.path.join("audio", filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    return filepath

# Helper: Transcribe using speech recognition
def transcribe_audio_chunks(wav_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = 20000
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            chunk.export(temp_wav.name, format="wav")
            with sr.AudioFile(temp_wav.name) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="ur-PK")
                    full_transcript += text + " "
                except sr.UnknownValueError:
                    full_transcript += "[Unrecognized] "
    return full_transcript.strip()

# Helper: Save to corpus
def save_corpus_file(dialect, filename, transcript):
    path = os.path.join("corpus", f"{dialect}_{filename}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return True

# Helper: Predict dialect (basic rule-based)
def predict_dialect(text):
    if any(word in text for word in ["وڈا", "ساڈا", "نئیں", "تہاڈا"]):
        return "Punjabi-Urdu"
    elif any(word in text for word in ["آہے", "پیو", "توھانجي", "چاٽو"]):
        return "Sindhi-Urdu"
    elif any(word in text for word in ["ئیے", "آ", "ہاں", "سا"]):
        return "Seraiki-Urdu"
    elif any(word in text for word in ["چہ", "کڑے", "دی", "زوڑ"]):
        return "Pashto-Urdu"
    elif any(word in text for word in ["ءَے", "کان", "گوں"]):
        return "Balochi-Urdu"
    elif any(word in text for word in ["ہیں", "کیا", "کیوں", "نہیں"]):
        return "Standard Urdu"
    return "Other/Unknown"

# Streamlit UI
st.title("🗺️ Digital Dialectal Mapper")

tabs = st.tabs(["📊 Corpus Overview", "🎤 Audio Input", "📁 Upload Audio"])

# -------- Tab 1: Corpus Overview --------
with tabs[0]:
    st.header("📊 Corpus Overview")
    data = []
    for fname in os.listdir("corpus"):
        dialect = fname.split("_")[0]
        with open(os.path.join("corpus", fname), "r", encoding="utf-8") as f:
            text = f.read()
        word_count = len(text.split())
        data.append({"Dialect": dialect, "File": fname, "Words": word_count, "Preview": text[:100]})
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No corpus files available.")

# -------- Tab 2: Audio Input (Record) --------
with tabs[1]:
    st.header("🎤 Record Your Voice")
    dialect_label = st.selectbox("Select dialect label (optional for training):", [
        "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu", "Pashto-Urdu", "Balochi-Urdu", "Standard Urdu", "Other/Unknown"
    ])
    audio_bytes = st_audiorec()
    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        st.success("Audio recorded.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            wav_path = tmp.name
        transcript = transcribe_audio_chunks(wav_path)
        st.subheader("📝 Transcription")
        st.write(transcript)
        predicted = predict_dialect(transcript)
        st.markdown(f"**Predicted Dialect:** 🧠 `{predicted}`")
        if st.button("💾 Save Recording"):
            audio_path = save_audio_file(audio_bytes, dialect_label)
            save_corpus_file(dialect_label, os.path.basename(audio_path), transcript)
            st.success("Recording and transcription saved.")

# -------- Tab 3: Audio Upload --------
with tabs[2]:
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    upload_label = st.selectbox("Select dialect label for upload:", [
        "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu", "Pashto-Urdu", "Balochi-Urdu", "Standard Urdu", "Other/Unknown"
    ])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            wav_path = tmp.name
        transcript = transcribe_audio_chunks(wav_path)
        st.subheader("📝 Transcription")
        st.write(transcript)
        predicted = predict_dialect(transcript)
        st.markdown(f"**Predicted Dialect:** 🧠 `{predicted}`")
        if st.button("💾 Save Upload"):
            saved_path = save_audio_file(open(wav_path, "rb").read(), upload_label)
            save_corpus_file(upload_label, os.path.basename(saved_path), transcript)
            st.success("Uploaded audio and transcription saved.")
