import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

# Config
CSV_PATH = "dialect_samples_extended.csv"
AUDIO_FOLDER = "audio"

# Required CSV columns
REQUIRED_COLUMNS = {
    'Region', 'Dialect Cluster', 'Latitude', 'Longitude',
    'Example Phrase', 'Audio File'
}

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        st.error(f"❗ CSV file must contain these columns: {REQUIRED_COLUMNS}")
        return None
    return df

def create_map(df):
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
    for _, row in df.iterrows():
        popup = f"""
        <b>{row['Dialect Cluster']}</b><br>
        Phrase: {row['Example Phrase']}<br>
        Region: {row['Region']}
        """
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup,
            tooltip=row['Dialect Cluster'],
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)
    return m

def play_audio(file_name):
    if pd.isna(file_name) or not isinstance(file_name, str) or not file_name.strip():
        st.info("No audio file available.")
        return

    full_path = os.path.join(AUDIO_FOLDER, file_name)
    if os.path.exists(full_path):
        audio_format = "audio/m4a" if file_name.lower().endswith(".m4a") else "audio/mp3"
        with open(full_path, "rb") as f:
            st.audio(f.read(), format=audio_format)
    else:
        st.warning(f"🔇 Audio file not found: {file_name}")

# App layout
st.set_page_config(page_title="Urdu Dialect Mapping Tool", layout="wide")
st.title("🗺️ Urdu Dialect Mapping and Profiling System")
st.markdown("Visualizes regional Urdu dialects with linguistic examples and recorded speech samples.")

# Load and validate data
if not os.path.exists(CSV_PATH):
    st.error(f"❌ File not found: {CSV_PATH}")
else:
    df = load_data(CSV_PATH)
    if df is not None:
        # Display map
        st.subheader("🧭 Regional Dialect Map")
        map_component = create_map(df)
        st_folium(map_component, width=1150, height=550)

        # Display table and audio
        st.subheader("🔉 Dialect Examples with Audio")
        for idx, row in df.iterrows():
            st.markdown(f"**Dialect Cluster:** {row['Dialect Cluster']}")
            st.markdown(f"📍 *Region:* {row['Region']}")
            st.markdown(f"📝 *Phrase:* {row['Example Phrase']}")
            play_audio(row.get("Audio File", ""))
            st.markdown("---")

        # Show full table
        with st.expander("📊 View Full Data Table"):
            st.dataframe(df, use_container_width=True)
