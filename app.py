import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import base64

st.set_page_config(page_title="Urdu Dialect Mapping", layout="wide")

st.title("🗺️ Digital Urdu Dialect Mapping and Profiling System")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("dialect_samples_extended.csv")
    return df

df = load_data()

required_columns = {
    "Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude",
    "Audio File", "Morphological Tag", "Semantic Feature", "Phonetic Variation", "Syntactic Structure"
}

# Validate columns
if not required_columns.issubset(set(df.columns)):
    st.error(f"❌ CSV is missing required columns.\n\nExpected: {required_columns}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Dialect Samples")

regions = st.sidebar.multiselect("Select Region(s)", options=sorted(df["Region"].unique()), default=df["Region"].unique())
clusters = st.sidebar.multiselect("Select Dialect Cluster(s)", options=sorted(df["Dialect Cluster"].unique()), default=df["Dialect Cluster"].unique())

filtered_df = df[(df["Region"].isin(regions)) & (df["Dialect Cluster"].isin(clusters))]

# Display Map
st.subheader("🗺️ Dialect Distribution Map")

m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

for _, row in filtered_df.iterrows():
    popup_html = f"""
    <b>Example:</b> {row['Example Phrase']}<br>
    <b>Dialect:</b> {row['Dialect Cluster']}<br>
    <b>Region:</b> {row['Region']}<br>
    <b>Phonetic Variation:</b> {row['Phonetic Variation']}<br>
    <b>Morphological Tag:</b> {row['Morphological Tag']}<br>
    <b>Semantic Feature:</b> {row['Semantic Feature']}<br>
    <b>Syntactic Structure:</b> {row['Syntactic Structure']}<br>
    """

    # Embed audio
    try:
        audio_file_path = row["Audio File"]
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode()
            audio_html = f"""
            <audio controls>
                <source src="data:audio/mp4;base64,{audio_base64}" type="audio/mp4">
                Your browser does not support the audio element.
            </audio>
            """
            popup_html += audio_html
    except Exception as e:
        popup_html += "<i>Audio not available</i>"

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(marker_cluster)

st_folium(m, width=1200, height=600)

# Show table of dialect data
st.subheader("📋 Dialect Data Table")

st.dataframe(
    filtered_df[
        [
            "Example Phrase",
            "Dialect Cluster",
            "Region",
            "Phonetic Variation",
            "Morphological Tag",
            "Semantic Feature",
            "Syntactic Structure",
        ]
    ].reset_index(drop=True),
    use_container_width=True,
)

# Optional: Audio preview player in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("▶️ Audio Sample Preview")

sample_row = filtered_df.sample(1).iloc[0]
st.sidebar.write(f"**Sample:** {sample_row['Example Phrase']} ({sample_row['Dialect Cluster']})")

try:
    st.sidebar.audio(sample_row["Audio File"], format="audio/m4a")
except:
    st.sidebar.write("_Audio file not found or unsupported format._")
