Code for Dialect Mapping
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
import zipfile
from pathlib import Path
import tempfile
import base64

# Optional audio dependencies - graceful fallback if not available
AUDIO_AVAILABLE = True
try:
    import speech_recognition as sr
    import wave
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    from st_audiorec import st_audiorec
except ImportError as e:
    AUDIO_AVAILABLE = False
    st.warning("⚠️ Audio processing libraries not found. Audio features will be disabled.")
    st.info("To enable audio features, install: pip install speech-recognition pydub st-audiorec")

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

# Audio Processing Functions (only if audio libraries are available)
if AUDIO_AVAILABLE:
    def convert_audio_to_wav(audio_file, target_sample_rate=16000):
        """Convert various audio formats to WAV format suitable for speech recognition."""
        try:
            # Load audio file using pydub
            audio = AudioSegment.from_file(audio_file)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            audio = audio.set_frame_rate(target_sample_rate)
            
            # Export to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                return tmp_file.name
        except Exception as e:
            st.error(f"Error converting audio: {e}")
            return None

    def transcribe_audio_chunks(audio_path, language='ur-PK'):
        """Transcribe audio by splitting it into chunks to handle longer files."""
        try:
            # Load the audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Split audio on silence to get chunks
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # 1 second of silence
                silence_thresh=audio.dBFS - 14,
                keep_silence=500,  # Keep 0.5 second of silence
                seek_step=1
            )
            
            # If no chunks found, use the whole audio
            if not chunks:
                chunks = [audio]
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            full_transcript = ""
            
            for i, chunk in enumerate(chunks):
                # Export chunk to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                    chunk.export(chunk_file.name, format="wav")
                    
                    # Transcribe chunk
                    try:
                        with sr.AudioFile(chunk_file.name) as source:
                            audio_data = recognizer.record(source)
                            
                        # Try Google Speech Recognition first (supports Urdu)
                        try:
                            text = recognizer.recognize_google(audio_data, language=language)
                            full_transcript += text + " "
                        except sr.UnknownValueError:
                            # Try with English if Urdu fails
                            try:
                                text = recognizer.recognize_google(audio_data, language='en-US')
                                full_transcript += f"[EN: {text}] "
                            except sr.UnknownValueError:
                                full_transcript += f"[Chunk {i+1}: Inaudible] "
                        except sr.RequestError as e:
                            st.warning(f"Could not request results from speech recognition service: {e}")
                            full_transcript += f"[Chunk {i+1}: Recognition failed] "
                    
                    except Exception as e:
                        full_transcript += f"[Chunk {i+1}: Error - {str(e)}] "
                    
                    # Clean up temporary file
                    try:
                        os.unlink(chunk_file.name)
                    except:
                        pass
            
            return full_transcript.strip()
        
        except Exception as e:
            st.error(f"Error in audio transcription: {e}")
            return None

    def transcribe_audio_simple(audio_path, language='ur-PK'):
        """Simple transcription for shorter audio files."""
        try:
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = recognizer.record(source)
            
            # Try multiple recognition methods
            transcript = ""
            
            # Method 1: Google Speech Recognition (best for Urdu)
            try:
                transcript = recognizer.recognize_google(audio_data, language=language)
                return transcript
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass
            
            # Method 2: Try with English
            try:
                transcript = recognizer.recognize_google(audio_data, language='en-US')
                return f"[EN] {transcript}"
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass
            
            return "Could not transcribe audio"
        
        except Exception as e:
            st.error(f"Error in simple transcription: {e}")
            return None

    def detect_language_from_audio_text(text):
        """Detect if the transcribed text is likely Urdu/Arabic script or English."""
        if not text:
            return "unknown"
        
        # Count Urdu/Arabic characters
        urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if urdu_chars > english_chars:
            return "urdu"
        elif english_chars > 0:
            return "english"
        else:
            return "unknown"

    def save_audio_file(audio_data, dialect, filename_prefix="audio"):
        """Save audio file to dialect-specific directory."""
        try:
            # Create audio directory for dialect
            audio_dir = Path(f"corpus/{dialect.replace(' ', '_').replace('-', '_')}/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.wav"
            file_path = audio_dir / filename
            
            # Save audio data
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            return str(file_path)
        except Exception as e:
            st.error(f"Error saving audio file: {e}")
            return None

else:
    # Placeholder functions when audio libraries are not available
    def convert_audio_to_wav(*args, **kwargs):
        st.error("Audio processing not available. Please install required libraries.")
        return None
    
    def transcribe_audio_chunks(*args, **kwargs):
        st.error("Audio transcription not available. Please install required libraries.")
        return None
    
    def transcribe_audio_simple(*args, **kwargs):
        st.error("Audio transcription not available. Please install required libraries.")
        return None
    
    def detect_language_from_audio_text(*args, **kwargs):
        return "unknown"
    
    def save_audio_file(*args, **kwargs):
        st.error("Audio saving not available. Please install required libraries.")
        return None

@st.cache_data
def load_corpus_index():
    """Load corpus index file that tracks all corpus files."""
    index_path = "corpus_index.json"
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading corpus index: {e}")
    return {}

def save_corpus_index(index_data):
    """Save corpus index to file."""
    try:
        with open("corpus_index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving corpus index: {e}")
        return False

def create_corpus_directory(dialect):
    """Create directory for dialect corpus if it doesn't exist."""
    corpus_dir = Path(f"corpus/{dialect.replace(' ', '_').replace('-', '_')}")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    return corpus_dir

def save_corpus_file(dialect, filename, content, metadata=None):
    """Save a corpus file with metadata."""
    try:
        corpus_dir = create_corpus_directory(dialect)
        file_path = corpus_dir / filename
        
        # Save the text content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Update corpus index
        index_data = load_corpus_index()
        if dialect not in index_data:
            index_data[dialect] = {}
        
        index_data[dialect][filename] = {
            "file_path": str(file_path),
            "upload_date": datetime.now().isoformat(),
            "word_count": len(tokenize(content)),
            "char_count": len(content),
            "metadata": metadata or {}
        }
        
        save_corpus_index(index_data)
        return True
    except Exception as e:
        st.error(f"Error saving corpus file: {e}")
        return False

def load_corpus_file(dialect, filename):
    """Load a specific corpus file."""
    try:
        index_data = load_corpus_index()
        if dialect in index_data and filename in index_data[dialect]:
            file_path = index_data[dialect][filename]["file_path"]
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        st.error(f"Error loading corpus file: {e}")
    return None

def get_corpus_statistics():
    """Get statistics for all dialect corpora."""
    index_data = load_corpus_index()
    stats = {}
    
    for dialect, files in index_data.items():
        total_texts = len(files)
        total_words = sum(file_info.get("word_count", 0) for file_info in files.values())
        total_chars = sum(file_info.get("char_count", 0) for file_info in files.values())
        
        stats[dialect] = {
            "total_texts": total_texts,
            "total_words": total_words,
            "total_chars": total_chars,
            "files": list(files.keys())
        }
    
    return stats

def search_corpus(dialect=None, query="", case_sensitive=False):
    """Search through corpus files."""
    results = []
    index_data = load_corpus_index()
    
    dialects_to_search = [dialect] if dialect else index_data.keys()
    
    for d in dialects_to_search:
        if d not in index_data:
            continue
            
        for filename, file_info in index_data[d].items():
            try:
                content = load_corpus_file(d, filename)
                if content:
                    if not case_sensitive:
                        content_search = content.lower()
                        query_search = query.lower()
                    else:
                        content_search = content
                        query_search = query
                    
                    if query_search in content_search:
                        # Find context around matches
                        matches = []
                        start = 0
                        while True:
                            pos = content_search.find(query_search, start)
                            if pos == -1:
                                break
                            
                            # Get context (50 chars before and after)
                            context_start = max(0, pos - 50)
                            context_end = min(len(content), pos + len(query) + 50)
                            context = content[context_start:context_end]
                            matches.append({
                                "position": pos,
                                "context": context
                            })
                            start = pos + 1
                        
                        if matches:
                            results.append({
                                "dialect": d,
                                "filename": filename,
                                "matches": len(matches),
                                "contexts": matches[:5]  # Limit to first 5 matches
                            })
            except Exception as e:
                st.error(f"Error searching file {filename}: {e}")
    
    return results

def analyze_corpus_linguistics(dialect):
    """Perform linguistic analysis on a dialect corpus."""
    index_data = load_corpus_index()
    if dialect not in index_data:
        return None
    
    all_text = ""
    file_count = 0
    
    # Combine all texts for the dialect
    for filename in index_data[dialect].keys():
        content = load_corpus_file(dialect, filename)
        if content:
            all_text += " " + content
            file_count += 1
    
    if not all_text.strip():
        return None
    
    # Tokenize and analyze
    tokens = tokenize(all_text)
    
    # Basic statistics
    word_freq = Counter(tokens)
    unique_words = len(word_freq)
    total_words = len(tokens)
    
    # Calculate lexical diversity (Type-Token Ratio)
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Most common words
    most_common = word_freq.most_common(20)
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
    
    # Sentence analysis (rough approximation)
    sentences = re.split(r'[۔؟!]', all_text)
    avg_sentence_length = np.mean([len(tokenize(sent)) for sent in sentences if sent.strip()]) if sentences else 0
    
    return {
        "file_count": file_count,
        "total_words": total_words,
        "unique_words": unique_words,
        "lexical_diversity": round(ttr, 4),
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "most_common_words": most_common,
        "total_characters": len(all_text)
    }

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

# Sidebar Dialect Legend and Corpus Management
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
    
    st.markdown("---")
    st.markdown("### 📚 Corpus Quick Stats")
    
    # Show corpus statistics in sidebar
    corpus_stats = get_corpus_statistics()
    for dialect, stats in corpus_stats.items():
        if stats['total_texts'] > 0:
            st.markdown(f"**{dialect}:** {stats['total_texts']} texts ({stats['total_words']} words)")
    
    if st.button("🔄 Refresh Corpus Stats"):
        st.rerun()

# Main navigation
main_tab = st.selectbox(
    "Select Main Section:",
    ["🏠 Home & Map", "📚 Corpus Management", "🎤 Audio Input", "📊 Analytics", "💬 Text Input"] if AUDIO_AVAILABLE else ["🏠 Home & Map", "📚 Corpus Management", "📊 Analytics", "💬 Text Input"],
    index=0
)

if main_tab == "🏠 Home & Map":
    # Home section with map
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

elif main_tab == "📚 Corpus Management":
    st.markdown("---")
    st.header("📚 Dialect Corpus Management")
    
    # Corpus management tabs
    corpus_tab1, corpus_tab2, corpus_tab3, corpus_tab4 = st.tabs

