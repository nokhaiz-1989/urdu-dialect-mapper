import streamlit as st
import pandas as pd
import plotly.express as px

# Set up the page
st.set_page_config(page_title="Urdu Dialect Mapping Tool", layout="wide")

# Title and description
st.title("📍 Digital Dialectal Mapping and Profiling System for Urdu")
st.markdown("""
This tool visualizes **Urdu dialect clusters** across regions of Pakistan.
You can explore regional phrases, filter dialects, and analyze geographic trends.
""")

# Load the CSV file
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dialect_samples_with_locations.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File not found. Please place 'dialect_samples_with_locations.csv' in the same folder or 'data' subfolder.")
        return pd.DataFrame()

df = load_data()

# Check if required columns are present
required_cols = {"Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude"}
if not required_cols.issubset(df.columns):
    st.error(f"❗ Your CSV file must contain these columns: {required_cols}")
    st.stop()

# Sidebar: Filter options
st.sidebar.header("🎯 Filter Dialect Clusters")
clusters = df["Dialect Cluster"].unique().tolist()
selected_clusters = st.sidebar.multiselect("Select Cluster(s):", options=clusters, default=clusters)

filtered_df = df[df["Dialect Cluster"].isin(selected_clusters)]

# Show filtered data
st.subheader("📋 Filtered Dialect Samples")
st.dataframe(filtered_df, use_container_width=True)

# Show pie chart
st.subheader("📊 Dialect Distribution")
pie = px.pie(
    filtered_df,
    names="Dialect Cluster",
    title="Distribution by Dialect Cluster",
    hole=0.4
)
st.plotly_chart(pie, use_container_width=True)

# Show interactive dialect map
st.subheader("🗺️ Geographical Mapping of Urdu Dialects")
map_fig = px.scatter_mapbox(
    filtered_df,
    lat="Latitude",
    lon="Longitude",
    color="Dialect Cluster",
    hover_name="Region",
    hover_data=["Example Phrase"],
    zoom=5,
    height=600
)
map_fig.update_layout(mapbox_style="open-street-map")
map_fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
st.plotly_chart(map_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📌 This tool is developed to assist in **linguistic profiling** and **regional dialect analysis** of Urdu across Pakistan.")
