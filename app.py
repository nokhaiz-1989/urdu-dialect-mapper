import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import Counter
from io import StringIO
import os
import re
import speech_recognition as sr

st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

# ---------- Load Data ----------
def load_data():
    file_path = "dialect_samples_extended.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=[
            "Dialect Cluster", "Example Phrase", "Region", "Latitude", "Longitude",
            "Morphological Tag", "Semantic Feature", "Phonetic Variation",
            "Syntactic Structure", "Audio File"
        ])

# ---------- Tokenizer ----------
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# ---------- Collocate Extractor ----------
def extract_collocates(df, dialect, keyword, window=2):
    phrases = df[df["Dialect Cluster"] == dialect]["Example Phrase"].dropna().tolist()
    collocates = Counter()
    for phrase in phrases:
        tokens = tokenize(phrase)
        for i, token in enumerate(tokens):
            if token == keyword:
                start = max(0, i - window)
                end = min(len(tokens), i + window + 1)
                context = tokens[start:i] + tokens[i+1:end]
                collocates.update(context)
    return collocates.most_common(10)

# ---------- Color by Dialect ----------
def assign_color(dialect):
    color_map = {
        'Sindhi-Urdu': 'red',
        'Punjabi-Urdu': 'blue',
        'Seraiki-Urdu': 'green',
        'Pashto-Urdu': 'orange',
        'Balochi-Urdu': 'purple',
        'Standard Urdu': 'darkblue'
    }
    return color_map.get(dialect, 'gray')

# ---------- Dialect Prediction ----------
def predict_dialect(text):
    text = text.lower()
    if any(word in text for word in ["توھان", "اچو", "چئو"]):
        return "Sindhi-Urdu"
    elif any(word in text for word in ["تساں", "وے", "ساڈا"]):
        return "Seraiki-Urdu"
    elif any(word in text for word in ["ساڈے", "نئیں", "اوہ"]):
        return "Punjabi-Urdu"
    elif any(word in text for word in ["کڑے", "زہ", "شو"]):
        return "Pashto-Urdu"
    elif any(word in text for word in ["چہ", "ہن"]):
        return "Balochi-Urdu"
    else:
        return "Standard Urdu"

# ---------- App Starts ----------
data = load_data()

st.markdown("---")
st.subheader("🗣️ Public User Text Input")

input_type = st.radio("Choose input type:", ["Written", "Spoken"], horizontal=True)
user_input = ""

if input_type == "Spoken":
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if st.button("🎙️ Record and Transcribe Audio"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording... Speak now")
            audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            st.session_state.transcription = recognizer.recognize_google(audio, language="ur-PK")
            st.success("Transcription successful")
        except sr.UnknownValueError:
            st.session_state.transcription = ""
            st.error("Could not understand the audio")
        except sr.RequestError:
            st.session_state.transcription = ""
            st.error("Speech recognition service unavailable")
    user_input = st.text_area("Your transcribed Urdu text:", value=st.session_state.transcription, height=200)
else:
    user_input = st.text_area("Paste written Urdu text:", height=200)

# ---------- Dialect Prediction ----------
dialect_guess = ""
if st.checkbox("Predict dialect automatically"):
    dialect_guess = predict_dialect(user_input)
    st.info(f"Predicted dialect: **{dialect_guess}**")
else:
    dialect_guess = st.selectbox("Select dialect:", [
        "Standard Urdu", "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu",
        "Pashto-Urdu", "Balochi-Urdu"
    ])

# ---------- Save and Tag ----------
if st.button("📥 Submit Input"):
    if user_input.strip():
        st.success(f"Text submitted successfully as {input_type} input.")
        new_row = {
            "Dialect Cluster": dialect_guess,
            "Example Phrase": user_input,
            "Region": "User Submission",
            "Latitude": 30.3753,
            "Longitude": 69.3451,
            "Morphological Tag": "Pending",
            "Semantic Feature": "Pending",
            "Phonetic Variation": "Pending",
            "Syntactic Structure": "Pending",
            "Audio File": None
        }
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

        feature_table = pd.DataFrame([{
            "Dialect": dialect_guess,
            "Phonetic Feature (Accent/Prosody)": "Pending Analysis",
            "Phonological Shift/Variants": "Pending",
            "Morphological Variation": "Pending",
            "Semantic Feature": "Pending"
        }])
        st.markdown("### 🧬 Preliminary Linguistic Feature Analysis")
        st.dataframe(feature_table)
    else:
        st.warning("⚠️ Please enter some text.")

# ---------- Map ----------
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
for _, row in data.iterrows():
    color = assign_color(row["Dialect Cluster"])
    popup = f"""
        <b>Dialect:</b> {row['Dialect Cluster']}<br>
        <b>Phrase:</b> {row['Example Phrase']}<br>
        <b>Region:</b> {row['Region']}<br>
        <b>Morph:</b> {row['Morphological Tag']}<br>
        <b>Semantic:</b> {row['Semantic Feature']}<br>
        <b>Phonetic:</b> {row['Phonetic Variation']}<br>
    """
    folium.Circle(
        location=[row["Latitude"], row["Longitude"]],
        radius=20000,
        color=color,
        fill=True,
        fill_opacity=0.2,
        fill_color=color
    ).add_to(m)
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=popup,
        icon=folium.Icon(color=color)
    ).add_to(m)

st.subheader("🗺️ Urdu Dialect Map")
st_folium(m, width=1000, height=600)

# ---------- Token Frequency ----------
st.subheader("📊 Token Frequency in Dialect")
dialect_options = data["Dialect Cluster"].dropna().unique().tolist()
selected_dialect = st.selectbox("Select a dialect:", ["All"] + dialect_options)

filtered_data = data.copy() if selected_dialect == "All" else data[data["Dialect Cluster"] == selected_dialect]
tokens = [token for phrase in filtered_data["Example Phrase"].dropna() for token in tokenize(phrase)]
token_counts = Counter(tokens).most_common(10)
st.dataframe(pd.DataFrame(token_counts, columns=["Token", "Frequency"]))

# ---------- Collocates ----------
keyword = st.text_input("🔍 Enter keyword for collocate analysis")
if keyword and selected_dialect != "All":
    collocates = extract_collocates(data, selected_dialect, keyword)
    st.subheader(f"Top Collocates with '{keyword}' in {selected_dialect}")
    st.dataframe(pd.DataFrame(collocates, columns=["Word", "Frequency"]))

# ---------- Full Table ----------
st.subheader("📄 Full Annotated Dataset")
st.dataframe(data.reset_index(drop=True), use_container_width=True)
