import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import os
import tempfile
import uuid
import pandas as pd

# App config
st.set_page_config(page_title="Digital Dialectal Mapper", layout="centered")
st.title("🗣️ Digital Dialectal Mapper for Urdu Dialects")

# Session state for table
if "data" not in st.session_state:
    st.session_state.data = []

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Temporary file save
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Convert to WAV if needed
    if tmp_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(tmp_path)
        wav_path = tmp_path.replace(".mp3", ".wav")
        sound.export(wav_path, format="wav")
    else:
        wav_path = tmp_path

    # Transcribe audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data, language="ur-PK")
            st.success("Transcription:")
            st.write(transcription)
        except sr.UnknownValueError:
            transcription = ""
            st.error("Could not understand audio.")
        except sr.RequestError as e:
            transcription = ""
            st.error(f"Speech Recognition error: {e}")

    # Dialect prediction (placeholder)
    def predict_dialect(text):
        if "تو" in text or "کدی" in text:
            return "Punjabi-Influenced Urdu"
        elif "چے" in text or "آھی" in text:
            return "Sindhi-Influenced Urdu"
        elif "ہے" in text or "ہوں" in text:
            return "Standard Urdu"
        else:
            return "Unknown Dialect"

    predicted_dialect = predict_dialect(transcription)
    st.success(f"Predicted Dialect: {predicted_dialect}")

    # Save result
    st.session_state.data.append({
        "ID": str(uuid.uuid4())[:8],
        "Transcription": transcription,
        "Predicted Dialect": predicted_dialect
    })

# Show results table
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.subheader("🔍 Transcriptions & Predictions")
    st.dataframe(df, use_container_width=True)
