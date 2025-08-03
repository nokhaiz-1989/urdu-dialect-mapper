import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(
    page_title="Digital Dialectal Mapper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Title
# ----------------------
st.title("🗺️ Digital Dialectal Mapper for Urdu Dialects")
st.markdown("Visualize and explore dialectal data of Urdu across different regions of Pakistan.")

# ----------------------
# Load GeoJSON
# ----------------------
@st.cache_data
def load_geojson():
    import json
    with open("dialect_regions.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return geojson_data

geojson_data = load_geojson()

# ----------------------
# Load Dialect Data CSV
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/dialect_samples.csv")
    return df

df = load_data()

# ----------------------
# Map
# ----------------------
st.subheader("🗺️ Urdu Dialect Map")
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

# Color mapping per dialect
dialect_colors = {
    "Punjabi-Urdu": "red",
    "Sindhi-Urdu": "green",
    "Pashto-Urdu": "blue",
    "Balochi-Urdu": "orange",
    "Standard Urdu": "purple"
}

# Add GeoJSON layer with color coding
folium.GeoJson(
    geojson_data,
    name="geojson",
    style_function=lambda feature: {
        "fillColor": dialect_colors.get(feature["properties"]["dialect"], "gray"),
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.5,
    },
    tooltip=folium.GeoJsonTooltip(fields=["name", "dialect"])
).add_to(m)

# Render the map
st_data = st_folium(m, width=1000, height=600)

# ----------------------
# Show CSV Table
# ----------------------
st.subheader("📋 Dialect Samples Table")
st.dataframe(df)
