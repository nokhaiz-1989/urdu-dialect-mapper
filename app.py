import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# CSV Path
CSV_PATH = "dialect_samples_with_locations.csv"
AUDIO_FOLDER = "audio"

# Required columns
REQUIRED_COLUMNS = {
    'Region', 'Dialect Cluster', 'Latitude', 'Longitude', 'Example Phrase'
}

# Load data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        st.error(f"❗ CSV file must contain these columns: {REQUIRED_COLUMNS}")
        return None
    return df

# Create folium map
def create_map(df):
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
    for _, row in df.iterrows():
        popup_text = f"<b>{row['Dialect Cluster']}</b><br>{row['Example Phrase']}<br><i>{row['Region']}</i>"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            tooltip=row['Dialect Cluster'],
            icon=folium.Icon(color="blue", icon="comment")
        ).add_to(m)
    return m

# Display audio player if audio file exists
def play_audio(audio_file):
    if isinstance(audio_file, str) and audio_file.strip():
        file_path_mp3 = os.path.join(AUDIO_FOLDER, audio_file)
        if os.path.exists(file_path_mp3):
            with open(file_path_mp3, "rb") as f:
                st.audio(f.read(), format="audio/m4a" if audio_file.endswith(".m4a") else "audio/mp3")
        else:
            st.warning("🔇 Audio file not found.")
    else:
        st.info("ℹ️ No audio file provided.")

# App UI
st.set_page_config(page_title="Urdu Dialect Mapper", layout="wide")
st.title("🗺️ Urdu Dialect Mapping Tool")
st.markdown("This tool visualizes Urdu dialect clusters across regions of Pakistan. You can explore regional phrases, hear dialectal speech, and analyze geographic trends.")

# Load CSV
if not os.path.exists(CSV_PATH):
    st.error(f"❌ File not found. Ensure '{CSV_PATH}' is in the correct folder.")
else:
    data = load_data(CSV_PATH)
    if data is not None:
        # Display the map
        st.subheader("🧭 Dialect Map of Pakistan")
        folium_map = create_map(data)
        st_folium(folium_map, width=1100, height=550)

        st.markdown("---")

        # Display dialect data with audio
        st.subheader("📋 Dialectal Data with Audio (if available)")
        for idx, row in data.iterrows():
            st.markdown(f"**{row['Dialect Cluster']}** – *{row['Region']}*")
            st.markdown(f"📝 *Phrase:* {row['Example Phrase']}")
            if 'Audio File' in row and pd.notna(row['Audio File']):
                play_audio(row['Audio File'])
            st.markdown("---")

        # Show full table
        with st.expander("🔍 See Full Dialect Table"):
            st.dataframe(data, use_container_width=True)
