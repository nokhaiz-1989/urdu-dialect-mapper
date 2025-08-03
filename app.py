import streamlit as st
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
import os
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

st.title("🗺️ Digital Dialectal Mapper for Urdu Dialects")
st.write("Upload an Urdu audio sample to transcribe it and predict the dialect.")

# Create session state for storing data
if "data" not in st.session_state:
    st.session_state.data = []

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=["mp3", "wav"])

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ur-PK")
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "API unavailable"

def predict_dialect(text):
    # Dummy rule-based dialect prediction logic
    text = text.lower()
    if any(word in text for word in ["توھان", "اچي", "وڃو"]):
        return "Sindhi-Urdu"
    elif any(word in text for word in ["ساڈے", "توہاڈے", "کیہہ"]):
        return "Punjabi-Urdu"
    elif any(word in text for word in ["تمھیں", "مجھے", "کیا"]):
        return "Standard Urdu"
    else:
        return "Unknown"

def get_dialect_location(dialect):
    locations = {
        "Sindhi-Urdu": [25.3969, 68.3578],    # Hyderabad
        "Punjabi-Urdu": [31.5204, 74.3587],   # Lahore
        "Standard Urdu": [33.6844, 73.0479],  # Islamabad
    }
    return locations.get(dialect, [30.3753, 69.3451])  # Default: Pakistan center

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save uploaded file
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert to WAV if needed
    if uploaded_file.type == "audio/mpeg":
        audio = AudioSegment.from_file("temp_audio", format="mp3")
        audio.export("converted.wav", format="wav")
        audio_path = "converted.wav"
    else:
        audio_path = "temp_audio"

    # Transcribe
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_path)
    
    st.subheader("📄 Transcription")
    st.success(transcription)

    # Predict dialect
    dialect = predict_dialect(transcription)
    st.subheader("🔍 Predicted Dialect")
    st.info(dialect)

    # Append to session data
    st.session_state.data.append({
        "Transcription": transcription,
        "Predicted Dialect": dialect
    })

    # Show map
    coords = get_dialect_location(dialect)
    st.subheader("🗺️ Dialect Location Map")
    map_ = folium.Map(location=coords, zoom_start=6)
    folium.Marker(coords, tooltip=dialect).add_to(map_)
    st_folium(map_, width=700)

# Show table
if st.session_state.data:
    st.subheader("🧾 All Samples Processed")
    df = pd.DataFrame(st.session_state.data)
    st.dataframe(df, use_container_width=True)
