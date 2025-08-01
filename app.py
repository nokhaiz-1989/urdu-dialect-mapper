import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import itertools

# Configuration
CSV_PATH = "dialect_samples_extended.csv"
AUDIO_FOLDER = "audio"
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

def get_dialect_colors(dialects):
    color_cycle = itertools.cycle([
        "red", "blue", "green", "purple", "orange",
        "darkred", "lightred", "beige", "darkblue",
        "darkgreen", "cadetblue", "darkpurple", "white",
        "pink", "lightblue", "lightgreen", "gray", "black"
    ])
    return {dialect: next(color_cycle) for dialect in sorted(dialects)}

def create_map(df):
    dialect_colors = get_dialect_colors(df["Dialect Cluster"].unique())
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

    for _, row in df.iterrows():
        popup_html = f"""
        <b>{row['Dialect Cluster']}</b><br>
        <i>{row['Example Phrase']}</i><br>
        Region: {row['Region']}
        """
        color = dialect_colors.get(row["Dialect Cluster"], "blue")
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_html,
            tooltip=row['Dialect Cluster'],
            icon=folium.Icon(color=color)
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

# Streamlit layout
st.set_page_config(page_title="Urdu Dialect Mapping Tool", layout="wide")
st.title("🗺️ Urdu Dialect Mapping and Profiling System")
st.markdown("Visualizes regional Urdu dialects with linguistic examples and recorded speech samples.")

# Load CSV
if not os.path.exists(CSV_PATH):
    st.error(f"❌ File not found: {CSV_PATH}")
else:
    df = load_data(CSV_PATH)
    if df is not None:
        # Map + Table layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📍 Interactive Map (Color-coded by Dialect)")
            map_component = create_map(df)
            st_folium(map_component, width=900, height=550)

        with col2:
            st.subheader("🔉 Play Dialect Samples")
            for idx, row in df.iterrows():
                st.markdown(f"**Dialect:** {row['Dialect Cluster']}")
                st.markdown(f"📍 *{row['Region']}* — _{row['Example Phrase']}_")
                play_audio(row['Audio File'])
                st.markdown("---")

        # Full table below with expandable option
        with st.expander("📊 View Full Data Table", expanded=False):
            st.dataframe(df, use_container_width=True)
