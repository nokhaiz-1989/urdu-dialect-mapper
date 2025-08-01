import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import Counter
import os
import re

# Page settings
st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

# Load dataset
def load_data():
    file_path = "dialect_samples_extended.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame(columns=[
        "Dialect Cluster", "Example Phrase", "Region", "Latitude", "Longitude",
        "Morphological Tag", "Semantic Feature", "Phonetic Variation",
        "Syntactic Structure", "Audio File"])

# Tokenization
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# Collocate extraction
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

# Dialect color mapping
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

# Dialect prediction from text
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

data = load_data()

# Sidebar legend
st.sidebar.markdown("## 🗺️ Dialect Legend")
for dialect in data["Dialect Cluster"].unique():
    color = assign_color(dialect)
    st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold;'>{dialect}</span>", unsafe_allow_html=True)

# User input
st.subheader("📝 Input Your Urdu Text")
user_input = st.text_area("Paste Urdu text here:", height=200)

if st.checkbox("Predict dialect automatically"):
    dialect_guess = predict_dialect(user_input)
    st.info(f"Predicted dialect: {dialect_guess}")
else:
    dialect_guess = st.selectbox("Or manually select dialect:", [
        "Standard Urdu", "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu",
        "Pashto-Urdu", "Balochi-Urdu"
    ])

if st.button("Submit Text"):
    if user_input.strip():
        st.success(f"Text submitted successfully for dialect: {dialect_guess}")
        new_entry = {
            "Dialect Cluster": dialect_guess,
            "Example Phrase": user_input,
            "Region": "User",
            "Latitude": 30.3753,
            "Longitude": 69.3451,
            "Morphological Tag": "Pending",
            "Semantic Feature": "Pending",
            "Phonetic Variation": "Pending",
            "Syntactic Structure": "Pending",
            "Audio File": None
        }
        data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)

# Map
st.subheader("🗺️ Urdu Dialect Map")
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

for _, row in data.iterrows():
    color = assign_color(row['Dialect Cluster'])
    folium.Circle(
        location=[row["Latitude"], row["Longitude"]],
        radius=30000,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.25
    ).add_to(m)

    html_popup = f"""
    <b>Dialect:</b> {row['Dialect Cluster']}<br>
    <b>Phrase:</b> {row['Example Phrase']}<br>
    <b>Region:</b> {row['Region']}<br>
    <b>Morph:</b> {row['Morphological Tag']}<br>
    <b>Semantic:</b> {row['Semantic Feature']}<br>
    <b>Phonetic:</b> {row['Phonetic Variation']}<br>
    <b>Syntactic:</b> {row['Syntactic Structure']}<br>
    """

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=folium.Popup(html_popup, max_width=300),
        icon=folium.Icon(color="gray")
    ).add_to(m)

st_folium(m, width=1000, height=600)

# Token Frequency
st.subheader("📊 Token Frequency")
dialects = data["Dialect Cluster"].dropna().unique()
selected_dialect = st.selectbox("Select Dialect", ["All"] + list(dialects))
filtered = data[data["Dialect Cluster"] == selected_dialect] if selected_dialect != "All" else data
tokens = [word for phrase in filtered["Example Phrase"].dropna() for word in tokenize(phrase)]
freq_df = pd.DataFrame(Counter(tokens).most_common(10), columns=["Token", "Frequency"])
st.write(freq_df)

# Collocates
keyword = st.text_input("🔎 Keyword for Collocate Analysis")
if keyword and selected_dialect != "All":
    st.subheader(f"Top Collocates with '{keyword}' in {selected_dialect}")
    collocates = extract_collocates(data, selected_dialect, keyword)
    st.write(pd.DataFrame(collocates, columns=["Word", "Frequency"]))

# Annotated Table
st.subheader("📋 Annotated Dataset")
st.dataframe(data.reset_index(drop=True), use_container_width=True)
