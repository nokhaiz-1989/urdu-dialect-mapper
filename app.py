import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import base64
from datetime import datetime

st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

st.title("🗺️ Digital Dialectal Mapper for Urdu Dialects")
st.markdown("Upload or record speech samples to map predicted dialects across Pakistan.")

# Load or initialize data
DATA_FILE = "dialect_samples.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Timestamp", "Transcription", "Predicted Dialect", "Latitude", "Longitude"])

# Dummy dialect prediction based on keywords
def predict_dialect(text):
    text = text.lower()
    if "aahe" in text or "tho" in text:
        return "Sindhi Urdu"
    elif "hain ji" in text or "nahi kara" in text:
        return "Punjabi Urdu"
    elif "kar raha hoon" in text:
        return "Standard Urdu"
    elif "balochi" in text:
        return "Balochi Urdu"
    else:
        return "Unknown"

# Record or upload audio
st.header("🎙️ Provide Audio Input")

audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        temp_path = tmp_file.name

    # Convert to suitable format (optional)
    audio = AudioSegment.from_file(temp_path)
    audio.export("converted.wav", format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile("converted.wav") as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data, language="ur-PK")
            st.success("Transcription: " + transcription)
        except sr.UnknownValueError:
            st.warning("Could not understand audio")
            transcription = ""
        except sr.RequestError as e:
            st.error(f"API Error: {e}")
            transcription = ""

    # Prediction and location
    predicted_dialect = predict_dialect(transcription)
    st.markdown(f"**Predicted Dialect:** `{predicted_dialect}`")

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=30.3753)
    with col2:
        lon = st.number_input("Longitude", value=69.3451)

    if st.button("➕ Add to Map"):
        df = pd.concat([
            df,
            pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Transcription": transcription,
                "Predicted Dialect": predicted_dialect,
                "Latitude": lat,
                "Longitude": lon
            }])
        ], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("Sample added successfully!")

# Map
st.header("🗺️ Dialect Map")

m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
for _, row in df.iterrows():
    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        tooltip=row["Predicted Dialect"],
        popup=f"{row['Transcription'][:100]}..."
    ).add_to(m)

st_data = st_folium(m, width=700, height=500)

# Table
st.header("📄 All Samples")
st.dataframe(df)

# Download
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.markdown(
    f'<a href="data:file/csv;base64,{b64}" download="dialect_samples._
