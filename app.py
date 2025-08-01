import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import Counter
import os
import re

# 🌐 Set page configuration
st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

# 📄 Load the CSV data
@st.cache_data
def load_data():
    path = "dialect_samples_extended.csv"
    return pd.read_csv(path)

# 🔤 Tokenizer for frequency analysis
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# 🔍 Collocate extractor
def extract_collocates(df, dialect, keyword, window=2):
    phrases = df[df["Dialect Cluster"] == dialect]["Example Phrase"].dropna().tolist()
    collocates = Counter()
    for phrase in phrases:
        tokens = tokenize(phrase)
        for i, token in enumerate(tokens):
            if token == keyword:
                context = tokens[max(0, i - window):i] + tokens[i + 1:i + 1 + window]
                collocates.update(context)
    return collocates.most_common(10)

# 🎨 Color assignment for dialects
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

# 🗂️ Load data
data = load_data()

# 📊 Sidebar: Filtering and Corpus Access
st.sidebar.title("🔍 Filter & Corpus Access")

dialect_options = data["Dialect Cluster"].dropna().unique().tolist()
selected_dialect = st.sidebar.selectbox("Select a Dialect", ["All"] + dialect_options)

keyword = st.sidebar.text_input("Collocate Keyword")

# 📚 Written Corpus Section
st.sidebar.markdown("---")
st.sidebar.header("📖 Dialect Corpus Viewer")

written_corpus_files = {
    "Standard Urdu": "standard_urdu_corpus.txt",
    # Add other corpora here
}
selected_written = st.sidebar.selectbox("Written Corpus", ["None"] + list(written_corpus_files.keys()))

spoken_corpus_files = {
    # Add spoken corpus mappings here
}
selected_spoken = st.sidebar.selectbox("Spoken Corpus", ["None"] + list(spoken_corpus_files.keys()))

# 🔄 Load selected corpus text
if selected_written != "None":
    path = os.path.join("corpora", written_corpus_files[selected_written])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.subheader(f"📘 Written Corpus: {selected_written}")
            st.text_area("Full Corpus Text", f.read(), height=300)
    else:
        st.warning(f"{selected_written} corpus file not found.")

if selected_spoken != "None":
    path = os.path.join("corpora", spoken_corpus_files[selected_spoken])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.subheader(f"🎙️ Spoken Corpus: {selected_spoken}")
            st.text_area("Full Spoken Corpus", f.read(), height=300)
    else:
        st.warning(f"{selected_spoken} spoken corpus file not found.")

# 📝 User Input for Text
st.markdown("---")
st.subheader("🗣️ Submit Your Dialectal Text")

input_type = st.radio("Input Type", ["Written", "Spoken"], horizontal=True)
user_input = st.text_area(f"Paste your {input_type.lower()} Urdu text here:", height=200)
dialect_guess = st.selectbox("Associate Dialect", [
    "Standard Urdu", "Lahori Urdu", "Karachi Urdu", "Peshawari Urdu", 
    "Quetta Urdu", "Seraiki-Urdu", "Sindhi-Urdu"
])

if st.button("Submit Text"):
    st.success(f"Text submitted as {input_type} input for {dialect_guess}.")
    st.markdown("### 🔬 Preliminary Linguistic Feature Analysis")
    feature_table = pd.DataFrame([{
        "Dialect": dialect_guess,
        "Phonetic Feature (Accent/Prosody)": "Pending Analysis",
        "Phonological Shift/Variants": "Pending Analysis",
        "Morphological Variation": "Pending Analysis",
        "Semantic Feature": "Pending Analysis"
    }])
    st.dataframe(feature_table)

# 📍 Data Filtering
filtered_data = data[data["Dialect Cluster"] == selected_dialect] if selected_dialect != "All" else data.copy()

# 🗺️ Create Map
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
for _, row in filtered_data.iterrows():
    color = assign_color(row['Dialect Cluster'])
    folium.Circle(
        location=[row["Latitude"], row["Longitude"]],
        radius=20000,
        color=color,
        fill=True,
        fill_opacity=0.2,
        fill_color=color
    ).add_to(m)

    # 📌 Marker with popup info
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
        icon=folium.Icon(color=color)
    ).add_to(m)

# 🌍 Display the map
st.subheader("🗺️ Dialect Map of Pakistan")
st_folium(m, width=1000, height=600)

# 📊 Token Frequency
st.subheader("📈 Token Frequency in Selected Dialect")
all_tokens = []
for phrase in filtered_data["Example Phrase"].dropna():
    all_tokens.extend(tokenize(phrase))
token_counts = Counter(all_tokens).most_common(10)
st.write(pd.DataFrame(token_counts, columns=["Token", "Frequency"]))

# 🔗 Collocates
if keyword:
    st.subheader(f"🔗 Collocates of '{keyword}' in {selected_dialect}")
    collocates = extract_collocates(data, selected_dialect, keyword)
    st.write(pd.DataFrame(collocates, columns=["Word", "Frequency"]))

# 📋 Display Full Table
st.subheader("📋 Full Annotated Dataset")
st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)
