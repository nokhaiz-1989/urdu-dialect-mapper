import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
import os
import json

# ✅ Set Streamlit page config FIRST
st.set_page_config(
    page_title="Digital Dialectal Mapper",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📍 Digital Dialectal Mapper")

# Load GeoJSON file
geojson_path = "dialect_regions.geojson"
if os.path.exists(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
else:
    st.error(f"GeoJSON file not found: {geojson_path}")
    st.stop()

# Load dialect sample CSV
csv_path = "data/dialect_samples.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.error(f"CSV file not found: {csv_path}")
    st.stop()

# Create Folium map
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5, tiles="CartoDB positron")

# Add GeoJSON overlay
folium.GeoJson(
    geojson_data,
    name="Dialects",
    tooltip=folium.GeoJsonTooltip(fields=["dialect"], aliases=["Dialect:"]),
    style_function=lambda feature: {
        'fillColor': '#3186cc',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5,
    },
).add_to(m)

# Render the map
st.subheader("🗺️ Map View")
st_data = st_folium(m, width=1200, height=500)

# Display dialect samples in table
st.subheader("📋 Dialect Sample Data")
st.dataframe(df)
