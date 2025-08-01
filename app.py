import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import Counter
from io import StringIO
import os
import re
import json

# Set the page configuration
st.set_page_config(page_title="Digital Dialectal Mapper", layout="wide")

# Sidebar Dialect Legend
st.sidebar.markdown("### 🗺️ Dialect Legend")
st.sidebar.markdown("""
- <span style='color:red'>Sindhi-Urdu</span>  
- <span style='color:blue'>Punjabi-Urdu</span>  
- <span style='color:green'>Seraiki-Urdu</span>  
- <span style='color:orange'>Pashto-Urdu</span>  
- <span style='color:purple'>Balochi-Urdu</span>  
- <span style='color:darkblue'>Standard Urdu</span>  
- <span style='color:gray'>Other/Unknown</span>
""", unsafe_allow_html=True)

# Load the CSV data
def load_data():
    path = "dialect_samples_extended.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=[
            "Dialect Cluster", "Example Phrase", "Region", "Latitude", "Longitude",
            "Morphological Tag", "Semantic Feature", "Phonetic Variation",
            "Syntactic Structure", "Audio File"])

# Load GeoJSON dialect region boundaries
def load_geojson():
    path = "dialect_regions.geojson"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

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
        'Seraiki-Urdu': 'green',
        'Pashto-Urdu': 'orange',
        'Balochi-Urdu': 'purple',
        'Standard Urdu': 'darkblue'
    }
    return color_map.get(dialect, 'gray')

# Mock dialect prediction from text
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

# Load and process data
data = load_data()
geojson_data = load_geojson()

# User input section
st.markdown("---")
st.subheader("\U0001F4AC Public User Text Input")

input_type = st.radio("Choose input type:", ["Written"], horizontal=True)
user_input = st.text_area("Paste your written Urdu text here:", height=200)

if st.checkbox("Predict dialect automatically from input OR select below:"):
    dialect_guess = predict_dialect(user_input)
    st.info(f"Predicted dialect: {dialect_guess}")
else:
    dialect_guess = st.selectbox("Select dialect to associate with input (optional):", ["Standard Urdu", "Lahori Urdu", "Karachi Urdu", "Peshawari Urdu", "Quetta Urdu", "Seraiki-Urdu", "Sindhi-Urdu"])

if st.button("Submit Text"):
    if user_input.strip():
        st.success(f"Text submitted successfully as {input_type} input for {dialect_guess} dialect.")
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
            "Phonological Shift/Variants": "Pending Analysis",
            "Morphological Variation": "Pending Analysis",
            "Semantic Feature": "Pending Analysis"
        }])
        st.markdown("### \U0001F50D Preliminary Linguistic Feature Analysis")
        st.dataframe(feature_table)
    else:
        st.warning("Please provide input text.")

# Map Initialization
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
selected_dialect_map = st.selectbox("Select a Dialect to Highlight on Map:", ["All"] + sorted(data["Dialect Cluster"].dropna().unique()))

# Show dialect regions using GeoJSON
if geojson_data:
    for feature in geojson_data["features"]:
        dialect = feature["properties"].get("dialect", "Unknown")
        color = assign_color(dialect)
        if selected_dialect_map == "All" or selected_dialect_map == dialect:
            folium.GeoJson(
                feature,
                name=dialect,
                style_function=lambda f, c=color: {
                    'fillColor': c,
                    'color': c,
                    'weight': 2,
                    'fillOpacity': 0.5
                }
            ).add_to(m)

folium.LayerControl().add_to(m)

st.subheader("\U0001F5FA Urdu Dialect Map")
st_folium(m, width=1000, height=600)

# Token Frequency
st.subheader("\U0001F4CA Token Frequency in Dialect")
dialect_options = data["Dialect Cluster"].dropna().unique().tolist()
selected_dialect = st.selectbox("Select a Dialect for Analysis", ["All"] + dialect_options)
filtered_data = data[data["Dialect Cluster"] == selected_dialect] if selected_dialect != "All" else data.copy()

all_tokens = []
for phrase in filtered_data["Example Phrase"].dropna():
    all_tokens.extend(tokenize(phrase))
token_counts = Counter(all_tokens).most_common(10)
st.write(pd.DataFrame(token_counts, columns=["Token", "Frequency"]))

# Collocates
keyword = st.text_input("Enter a keyword for collocate analysis")
if keyword and selected_dialect != "All":
    st.subheader(f"\U0001F50D Top Collocates with '{keyword}' in {selected_dialect}")
    collocates = extract_collocates(data, selected_dialect, keyword)
    st.write(pd.DataFrame(collocates, columns=["Word", "Frequency"]))

# Raw Table
st.subheader("\U0001F4CB Complete Annotated Dataset")
st.dataframe(data.reset_index(drop=True), use_container_width=True)
