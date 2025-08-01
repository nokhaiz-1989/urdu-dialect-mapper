import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from pathlib import Path

st.set_page_config(page_title="Urdu Dialect Mapper", layout="wide")
st.title("📍 Urdu Dialect Mapping and Profiling System")

# --- Load CSV Data ---
CSV_PATH = "dialect_samples_with_locations.csv"
AUDIO_FOLDER = "audio"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    expected_cols = {"Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude"}
    if not expected_cols.issubset(df.columns):
        st.error(f"CSV file must contain these columns: {expected_cols}")
        st.stop()
    return df

data = load_data(CSV_PATH)

# --- Create Map ---
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5, control_scale=True)

for _, row in data.iterrows():
    popup_html = f"""
        <strong>{row['Example Phrase']}</strong><br>
        Dialect: {row['Dialect Cluster']}<br>
        Region: {row['Region']}<br>
        <audio controls>
            <source src="{row['Audio File']}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

with st.container():
    folium_static(m, height=600)

# --- Upload New Audio (Placeholder for Future ML Integration) ---
st.subheader("🔊 Upload Audio to Detect Dialect")
uploaded_file = st.file_uploader("Upload an Urdu audio clip (.mp3)", type=["mp3"])
if uploaded_file:
    st.audio(uploaded_file, format='audio/mp3')
    st.success("✅ File uploaded. (Dialect detection coming soon!)")
    # Placeholder for transcription + matching

# --- Data Table ---
st.subheader("📊 Annotated Dialectal Samples")
with st.expander("View Full Table"):
    st.dataframe(data, use_container_width=True)
