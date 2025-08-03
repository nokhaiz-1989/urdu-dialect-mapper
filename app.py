import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from collections import Counter
import os
import re
import json
import numpy as np
from datetime import datetime

# Set the page configuration
st.set_page_config(
    page_title="Digital Dialectal Mapper", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dialect-legend {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🗺️ Digital Dialectal Mapper</h1>', unsafe_allow_html=True)

# Sidebar Dialect Legend
with st.sidebar:
    st.markdown("### 🗺️ Dialect Legend")
    st.markdown("""
    <div class="dialect-legend">
    • <span style='color:red'>■</span> Sindhi-Urdu<br>
    • <span style='color:blue'>■</span> Punjabi-Urdu<br>
    • <span style='color:green'>■</span> Seraiki-Urdu<br>
    • <span style='color:orange'>■</span> Pashto-Urdu<br>
    • <span style='color:purple'>■</span> Balochi-Urdu<br>
    • <span style='color:darkblue'>■</span> Standard Urdu<br>
    • <span style='color:gray'>■</span> Other/Unknown
    </div>
    """, unsafe_allow_html=True)

# Enhanced data loading with error handling and caching
@st.cache_data
def load_data():
    """Load CSV data with error handling and default fallback."""
    path = "dialect_samples_extended.csv"
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure required columns exist
            required_columns = [
                "Dialect Cluster", "Example Phrase", "Region", "Latitude", "Longitude",
                "Morphological Tag", "Semantic Feature", "Phonetic Variation",
                "Syntactic Structure", "Audio File"
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            return df
        else:
            st.warning(f"CSV file not found at {path}. Using sample data.")
            return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data if CSV is not available."""
    sample_data = [
        {"Dialect Cluster": "Punjabi-Urdu", "Example Phrase": "تساں کتھے جانا اے", "Region": "Lahore", 
         "Latitude": 31.5204, "Longitude": 74.3587, "Morphological Tag": "Question", 
         "Semantic Feature": "Direction", "Phonetic Variation": "ت aspirated", 
         "Syntactic Structure": "SVO", "Audio File": None},
        {"Dialect Cluster": "Sindhi-Urdu", "Example Phrase": "توھان ڪيئن آھيو", "Region": "Karachi", 
         "Latitude": 24.8607, "Longitude": 67.0011, "Morphological Tag": "Greeting", 
         "Semantic Feature": "Politeness", "Phonetic Variation": "ڪ velar", 
         "Syntactic Structure": "SOV", "Audio File": None},
        {"Dialect Cluster": "Pashto-Urdu", "Example Phrase": "ته چېرته ځې", "Region": "Peshawar", 
         "Latitude": 34.0151, "Longitude": 71.5249, "Morphological Tag": "Question", 
         "Semantic Feature": "Direction", "Phonetic Variation": "ځ retroflex", 
         "Syntactic Structure": "SOV", "Audio File": None},
    ]
    return pd.DataFrame(sample_data)

@st.cache_data
def load_geojson():
    """Load GeoJSON data with error handling."""
    path = "dialect_regions.geojson"
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
    return None

def tokenize(text):
    """Enhanced tokenizer with better Unicode support."""
    if pd.isna(text):
        return []
    # Handle Urdu text better
    return re.findall(r'[\u0600-\u06FF\w]+', str(text).lower())

def extract_collocates(df, dialect, keyword, window=2):
    """Extract collocates with improved error handling."""
    try:
        phrases = df[df["Dialect Cluster"] == dialect]["Example Phrase"].dropna().tolist()
        collocates = Counter()
        keyword = keyword.lower().strip()
        
        for phrase in phrases:
            tokens = tokenize(phrase)
            for i, token in enumerate(tokens):
                if token == keyword:
                    start = max(0, i - window)
                    end = min(len(tokens), i + window + 1)
                    context = tokens[start:i] + tokens[i+1:end]
                    collocates.update(context)
        return collocates.most_common(10)
    except Exception as e:
        st.error(f"Error in collocate extraction: {e}")
        return []

def assign_color(dialect):
    """Assign colors to dialects with consistent mapping."""
    color_map = {
        'Sindhi-Urdu': '#FF0000',      # Red
        'Punjabi-Urdu': '#0000FF',     # Blue
        'Seraiki-Urdu': '#008000',     # Green
        'Pashto-Urdu': '#FFA500',      # Orange
        'Balochi-Urdu': '#800080',     # Purple
        'Standard Urdu': '#000080',    # Dark Blue
        'Lahori Urdu': '#4169E1',      # Royal Blue
        'Karachi Urdu': '#DC143C',     # Crimson
        'Peshawari Urdu': '#FF8C00',   # Dark Orange
        'Quetta Urdu': '#9932CC'       # Dark Orchid
    }
    return color_map.get(dialect, '#808080')  # Gray for unknown

def predict_dialect(text):
    """Enhanced dialect prediction with more patterns."""
    if not text or pd.isna(text):
        return "Standard Urdu"
    
    text = text.lower().strip()
    
    # Sindhi patterns
    sindhi_patterns = ["توھان", "اچو", "چئو", "ڪيئن", "آھيو", "ڪري"]
    if any(word in text for word in sindhi_patterns):
        return "Sindhi-Urdu"
    
    # Seraiki patterns
    seraiki_patterns = ["تساں", "وے", "ساڈا", "کھڑے", "جانا اے"]
    if any(word in text for word in seraiki_patterns):
        return "Seraiki-Urdu"
    
    # Punjabi patterns
    punjabi_patterns = ["ساڈے", "نئیں", "اوہ", "کی گل", "تسی"]
    if any(word in text for word in punjabi_patterns):
        return "Punjabi-Urdu"
    
    # Pashto patterns
    pashto_patterns = ["کڑے", "زہ", "شو", "چېرته", "ځې"]
    if any(word in text for word in pashto_patterns):
        return "Pashto-Urdu"
    
    # Balochi patterns
    balochi_patterns = ["چہ", "ہن", "کنت", "گون"]
    if any(word in text for word in balochi_patterns):
        return "Balochi-Urdu"
    
    return "Standard Urdu"

def create_heat_map_data(df):
    """Create heat map data from dialect distribution."""
    try:
        # Group by coordinates and count occurrences
        heat_data = []
        for _, row in df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                heat_data.append([
                    float(row['Latitude']), 
                    float(row['Longitude']), 
                    1  # Weight - can be adjusted based on frequency
                ])
        return heat_data
    except Exception as e:
        st.error(f"Error creating heat map data: {e}")
        return []

def create_dialect_density_data(df):
    """Create density data for each dialect."""
    density_data = {}
    for dialect in df['Dialect Cluster'].dropna().unique():
        dialect_df = df[df['Dialect Cluster'] == dialect]
        dialect_heat_data = []
        for _, row in dialect_df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                dialect_heat_data.append([
                    float(row['Latitude']), 
                    float(row['Longitude']), 
                    1
                ])
        density_data[dialect] = dialect_heat_data
    return density_data

# Load data
data = load_data()
geojson_data = load_geojson()

# User input section
st.markdown("---")
st.subheader("💬 Public User Text Input")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Paste your written Urdu text here:", 
        height=150,
        placeholder="یہاں اردو متن لکھیں..."
    )

with col2:
    input_type = st.radio("Input type:", ["Written"], horizontal=True)
    
    auto_predict = st.checkbox("🤖 Auto-predict dialect")
    
    if auto_predict and user_input.strip():
        dialect_guess = predict_dialect(user_input)
        st.success(f"Predicted: **{dialect_guess}**")
    else:
        dialect_options = [
            "Standard Urdu", "Lahori Urdu", "Karachi Urdu", 
            "Peshawari Urdu", "Quetta Urdu", "Seraiki-Urdu", "Sindhi-Urdu"
        ]
        dialect_guess = st.selectbox("Select dialect:", dialect_options)

if st.button("📤 Submit Text", type="primary"):
    if user_input.strip():
        # Add timestamp for better tracking
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            "Dialect Cluster": dialect_guess,
            "Example Phrase": user_input.strip(),
            "Region": "User Submission",
            "Latitude": 30.3753,  # Default Pakistan center
            "Longitude": 69.3451,
            "Morphological Tag": "Pending",
            "Semantic Feature": "Pending",
            "Phonetic Variation": "Pending",
            "Syntactic Structure": "Pending",
            "Audio File": None,
            "Timestamp": timestamp
        }
        
        # Add to dataframe
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        
        st.success(f"✅ Text submitted successfully as {dialect_guess} dialect!")
        
        # Show preliminary analysis
        with st.expander("🔍 Preliminary Linguistic Analysis"):
            feature_table = pd.DataFrame([{
                "Dialect": dialect_guess,
                "Text Length": len(user_input),
                "Word Count": len(tokenize(user_input)),
                "Phonetic Feature": "Pending Analysis",
                "Morphological Variation": "Pending Analysis",
                "Semantic Feature": "Pending Analysis"
            }])
            st.dataframe(feature_table, use_container_width=True)
    else:
        st.warning("⚠️ Please provide input text.")

# Map visualization section
st.markdown("---")
st.subheader("🗺️ Interactive Dialect Map")

# Map type selection
map_type = st.radio(
    "Select map visualization:",
    ["Regional Boundaries", "Heat Map", "Dialect Density"],
    horizontal=True
)

# Dialect filter
col1, col2 = st.columns([1, 1])
with col1:
    selected_dialect_map = st.selectbox(
        "Filter by dialect:", 
        ["All"] + sorted(data["Dialect Cluster"].dropna().unique())
    )

with col2:
    if map_type == "Heat Map":
        heat_radius = st.slider("Heat map radius:", 10, 50, 25)
        heat_blur = st.slider("Heat map blur:", 5, 25, 15)

# Create map
m = folium.Map(
    location=[30.3753, 69.3451], 
    zoom_start=5,
    tiles='OpenStreetMap'
)

# Filter data based on selection
if selected_dialect_map != "All":
    filtered_map_data = data[data["Dialect Cluster"] == selected_dialect_map]
else:
    filtered_map_data = data.copy()

# Add visualizations based on type
if map_type == "Regional Boundaries":
    # Show dialect regions using GeoJSON
    if geojson_data:
        for feature in geojson_data["features"]:
            dialect = feature["properties"].get("dialect", "Unknown")
            color = assign_color(dialect)
            if selected_dialect_map == "All" or selected_dialect_map == dialect:
                folium.GeoJson(
                    feature,
                    name=dialect,
                    style_function=lambda feature, c=color: {
                        'fillColor': c,
                        'color': c,
                        'weight': 2,
                        'fillOpacity': 0.4,
                        'opacity': 0.8
                    },
                    tooltip=folium.Tooltip(dialect)
                ).add_to(m)
    
    # Add point markers
    for _, row in filtered_map_data.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=folium.Popup(
                    f"<b>{row['Dialect Cluster']}</b><br>"
                    f"Region: {row['Region']}<br>"
                    f"Phrase: {row['Example Phrase'][:50]}...",
                    max_width=200
                ),
                color=assign_color(row['Dialect Cluster']),
                fill=True,
                weight=2
            ).add_to(m)

elif map_type == "Heat Map":
    # Create overall heat map
    heat_data = create_heat_map_data(filtered_map_data)
    if heat_data:
        HeatMap(
            heat_data,
            radius=heat_radius,
            blur=heat_blur,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(m)

elif map_type == "Dialect Density":
    # Create separate heat maps for each dialect
    density_data = create_dialect_density_data(filtered_map_data)
    
    for dialect, heat_data in density_data.items():
        if heat_data and (selected_dialect_map == "All" or selected_dialect_map == dialect):
            # Use dialect-specific colors for heat maps
            color = assign_color(dialect)
            HeatMap(
                heat_data,
                name=f"{dialect} Density",
                radius=20,
                blur=15,
                gradient={0.2: color, 0.4: color, 0.6: color, 1: color},
                opacity=0.6
            ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
st_folium(m, width=1000, height=600)

# Analytics section
st.markdown("---")
st.subheader("📊 Dialect Analytics")

col1, col2 = st.columns(2)

with col1:
    # Token Frequency Analysis
    st.markdown("#### 📈 Token Frequency Analysis")
    dialect_options = data["Dialect Cluster"].dropna().unique().tolist()
    selected_dialect = st.selectbox("Select dialect for analysis:", ["All"] + dialect_options)
    
    filtered_data = data[data["Dialect Cluster"] == selected_dialect] if selected_dialect != "All" else data.copy()
    
    # Calculate token frequencies
    all_tokens = []
    for phrase in filtered_data["Example Phrase"].dropna():
        all_tokens.extend(tokenize(phrase))
    
    if all_tokens:
        token_counts = Counter(all_tokens).most_common(10)
        token_df = pd.DataFrame(token_counts, columns=["Token", "Frequency"])
        st.dataframe(token_df, use_container_width=True)
    else:
        st.info("No tokens found for the selected dialect.")

with col2:
    # Collocate Analysis
    st.markdown("#### 🔍 Collocate Analysis")
    keyword = st.text_input("Enter keyword for collocate analysis:", placeholder="کیا")
    
    if keyword and selected_dialect != "All":
        collocates = extract_collocates(data, selected_dialect, keyword)
        if collocates:
            collocate_df = pd.DataFrame(collocates, columns=["Word", "Frequency"])
            st.dataframe(collocate_df, use_container_width=True)
        else:
            st.info(f"No collocates found for '{keyword}' in {selected_dialect}")

# Dataset overview
st.markdown("---")
st.subheader("📋 Complete Dataset")
st.dataframe(data.reset_index(drop=True), use_container_width=True)

# Statistics
st.markdown("---")
st.subheader("📈 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", len(data))

with col2:
    st.metric("Unique Dialects", data["Dialect Cluster"].nunique())

with col3:
    st.metric("Unique Regions", data["Region"].nunique())

with col4:
    user_submissions = len(data[data["Region"] == "User Submission"])
    st.metric("User Submissions", user_submissions)

# Dialect distribution
if len(data) > 0:
    st.markdown("#### Dialect Distribution")
    dialect_counts = data["Dialect Cluster"].value_counts()
    st.bar_chart(dialect_counts)
