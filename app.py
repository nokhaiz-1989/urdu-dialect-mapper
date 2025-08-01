import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import Counter
from io import StringIO
import os
import re
import requests

# Set the page configuration
st.set_page_config(page_title="Urdu Dialect Mapper", layout="wide")

# Load the CSV data
def load_data():
    path = "dialect_samples_extended.csv"
    df = pd.read_csv(path)
    return df

# Tokenizer for simple word frequency
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# Collocate extractor
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

# Assign a color to each dialect
def assign_color(dialect):
    color_map = {
        'Sindhi-Urdu': 'red',
        'Punjabi-Urdu': 'blue',
        'Saraiki-Urdu': 'green',
        'Pashto-Urdu': 'orange',
        'Balochi-Urdu': 'purple',
        'Standard Urdu': 'darkblue'
    }
    return color_map.get(dialect, 'gray')

# Load and process data
data = load_data()

# Sidebar - Dialect filter
st.sidebar.title("Filter & Analysis")
dialect_options = data["Dialect Cluster"].dropna().unique().tolist()
selected_dialect = st.sidebar.selectbox("Select a Dialect", ["All"] + dialect_options)

# Sidebar - Collocate keyword
keyword = st.sidebar.text_input("Enter a keyword for collocate analysis")

# Sidebar - Dropbox link for corpus
st.sidebar.markdown("---")
st.sidebar.subheader("Dialect Corpus Uploader")
dropbox_link = st.sidebar.text_input("Paste Dropbox Text File Link")
corpus_tokens = []

if dropbox_link and dropbox_link.startswith("https://www.dropbox.com"):
    direct_link = dropbox_link.replace("www.dropbox.com", "dl.dropboxusercontent.com")
    try:
        response = requests.get(direct_link)
        if response.status_code == 200:
            text_data = response.text
            corpus_tokens = tokenize(text_data)
            st.sidebar.success("Corpus loaded and tokenized.")
        else:
            st.sidebar.error("Failed to fetch file from Dropbox.")
    except Exception as e:
        st.sidebar.error(f"Error fetching Dropbox file: {e}")

# Filter data
if selected_dialect != "All":
    filtered_data = data[data["Dialect Cluster"] == selected_dialect]
else:
    filtered_data = data.copy()

# Map Initialization
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

# Add markers with color
for _, row in filtered_data.iterrows():
    popup_html = f"""
    <b>Dialect:</b> {row['Dialect Cluster']}<br>
    <b>Region:</b> {row['Region']}<br>
    <b>Phrase:</b> {row['Example Phrase']}<br>
    <b>Morph:</b> {row['Morphological Tag']}<br>
    <b>Semantic:</b> {row['Semantic Feature']}<br>
    <b>Phonetic:</b> {row['Phonetic Variation']}<br>
    <b>Syntactic:</b> {row['Syntactic Structure']}<br>
    """
    if not pd.isna(row.get("Audio File")):
        popup_html += f'<audio controls src="audio/{row["Audio File"]}" type="audio/mpeg"></audio>'

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=assign_color(row['Dialect Cluster']))
    ).add_to(m)

# Display the map
st.subheader("🗺 Urdu Dialect Map")
st_folium(m, width=1000, height=550)

# Token Frequency
st.subheader("📊 Token Frequency in Dialect")
all_tokens = []
for phrase in filtered_data["Example Phrase"].dropna():
    all_tokens.extend(tokenize(phrase))
token_counts = Counter(all_tokens).most_common(10)
st.write(pd.DataFrame(token_counts, columns=["Token", "Frequency"]))

# Dropbox Corpus Token Frequency
if corpus_tokens:
    st.subheader("📚 Token Frequency from Dropbox Corpus")
    corpus_counts = Counter(corpus_tokens).most_common(10)
    st.write(pd.DataFrame(corpus_counts, columns=["Token", "Frequency"]))

# Collocates
if keyword:
    st.subheader(f"🔍 Top Collocates with '{keyword}' in {selected_dialect}")
    collocates = extract_collocates(data, selected_dialect, keyword)
    st.write(pd.DataFrame(collocates, columns=["Word", "Frequency"]))

# Raw Table
st.subheader("📋 Complete Annotated Dataset")
st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)

# Remove excess space below
st.markdown("<style>main > div:has(.block-container) > div:last-child {padding-bottom: 0px !important;}</style>", unsafe_allow_html=True)
