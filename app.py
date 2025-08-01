import streamlit as st
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(layout="wide")
st.title("🗺️ Urdu Dialect Mapping Tool")

# Sidebar for dialect selection
st.sidebar.header("🗣️ Select a Dialect")
selected_dialect = st.sidebar.selectbox("Choose a dialect to highlight:", [
    "All", "Karachi", "Lahori", "Peshawari", "Hyderabadi", "Seraiki", "Kohati", "Multani", "Mirpuri"
])

# Dialect colors (you can customize these)
dialect_colors = {
    "Karachi": "blue",
    "Lahori": "green",
    "Peshawari": "orange",
    "Hyderabadi": "purple",
    "Seraiki": "red",
    "Kohati": "pink",
    "Multani": "brown",
    "Mirpuri": "cadetblue"
}

# Legend
st.sidebar.markdown("### 🎨 Dialect Colors")
for dialect, color in dialect_colors.items():
    st.sidebar.markdown(f"<div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: {color}; margin-right: 10px;'></div>{dialect}</div>", unsafe_allow_html=True)

# Load GeoJSON with dialect regions
geojson_path = "urdu_dialects.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    dialect_geojson = json.load(f)

# Create map
m = folium.Map(location=[30.3753, 69.3451], zoom_start=6, tiles='OpenStreetMap')

# Add polygons to the map
for feature in dialect_geojson["features"]:
    region_dialect = feature["properties"]["dialect"]
    if selected_dialect == "All" or selected_dialect == region_dialect:
        color = dialect_colors.get(region_dialect, "gray")
        folium.GeoJson(
            feature,
            style_function=lambda x, color=color: {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.6,
            },
            tooltip=region_dialect
        ).add_to(m)

# Show the map
st_folium(m, width=1300, height=700)
