# --- Legend in Sidebar ---
st.sidebar.markdown("### 🗺️ Dialect Legend")
for dialect, color in dialect_colors.items():
    st.sidebar.markdown(f"<span style='color:{color}; font-weight:bold;'>{dialect}</span>", unsafe_allow_html=True)

# --- Dialect Selection for Highlighting ---
st.sidebar.markdown("---")
highlight_dialect = st.sidebar.selectbox("🎯 Highlight Only This Dialect", ["All"] + sorted(dialect_colors.keys()))

# --- Region Polygons on Map ---
m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)

for _, row in data.iterrows():
    dialect = row["Dialect Cluster"]
    if highlight_dialect != "All" and dialect != highlight_dialect:
        continue  # skip if not the selected dialect

    color = assign_color(dialect)

    # Create a square polygon around the point
    lat, lon = row["Latitude"], row["Longitude"]
    delta = 0.5  # degrees to make a square region
    bounds = [
        [lat - delta, lon - delta],
        [lat - delta, lon + delta],
        [lat + delta, lon + delta],
        [lat + delta, lon - delta],
    ]
    folium.Polygon(
        locations=bounds,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.4,
        weight=1
    ).add_to(m)

    # Popup Info
    popup_html = f"""
        <b>Dialect:</b> {dialect}<br>
        <b>Region:</b> {row['Region']}<br>
        <b>Phrase:</b> {row['Example Phrase']}<br>
        <b>Morph:</b> {row['Morphological Tag']}<br>
        <b>Semantic:</b> {row['Semantic Feature']}<br>
        <b>Phonetic:</b> {row['Phonetic Variation']}<br>
        <b>Syntactic:</b> {row['Syntactic Structure']}<br>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=color)
    ).add_to(m)

# --- Display Map ---
st.subheader("🗺️ Urdu Dialect Map")
st_folium(m, width=1000, height=600)
