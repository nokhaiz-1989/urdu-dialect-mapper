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
import speech_recognition as sr
import wave
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import base64
from st_audiorec import st_audiorec

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

# Audio Processing Functions
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
                        # Fallback to other recognition methods
                        try:
                            text = recognizer.recognize_sphinx(audio_data)
                            full_transcript += f"[Offline: {text}] "
                        except:
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
        
        # Method 3: Offline recognition (fallback)
        try:
            transcript = recognizer.recognize_sphinx(audio_data)
            return f"[Offline] {transcript}"
        except:
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

# Main navigation
main_tab = st.selectbox(
    "Select Main Section:",
    ["🏠 Home & Map", "📚 Corpus Management", "🎤 Audio Input", "📊 Analytics", "💬 Text Input"],
    index=0
)

if main_tab == "📚 Corpus Management":
    st.markdown("---")
    st.header("📚 Dialect Corpus Management")
    
    # Corpus management tabs
    corpus_tab1, corpus_tab2, corpus_tab3, corpus_tab4 = st.tabs([
        "📤 Upload Corpus", 
        "📋 Browse Corpus", 
        "🔍 Search Corpus", 
        "📈 Corpus Analytics"
    ])
    
    with corpus_tab1:
        st.subheader("📤 Upload Corpus Files")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dialect selection for corpus
            available_dialects = [
                "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu", 
                "Pashto-Urdu", "Balochi-Urdu", "Standard Urdu",
                "Lahori Urdu", "Karachi Urdu", "Peshawari Urdu", "Quetta Urdu"
            ]
            selected_corpus_dialect = st.selectbox("Select dialect for corpus:", available_dialects)
            
            # File upload options
            upload_method = st.radio("Upload method:", ["Single Text File", "Multiple Files", "Text Input"])
            
            if upload_method == "Single Text File":
                uploaded_file = st.file_uploader(
                    "Choose a text file", 
                    type=['txt', 'csv'],
                    help="Upload a single text file for the selected dialect"
                )
                
                if uploaded_file is not None:
                    try:
                        # Read file content
                        content = uploaded_file.read().decode('utf-8-sig')
                        
                        # Show preview
                        st.text_area("File preview:", content[:500] + "..." if len(content) > 500 else content, height=150)
                        
                        # Metadata input
                        st.markdown("**File Metadata (Optional):**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            source = st.text_input("Source (e.g., newspaper, book):", key="single_source")
                            author = st.text_input("Author/Speaker:", key="single_author")
                        with col_b:
                            genre = st.selectbox("Genre:", ["Literary", "News", "Conversational", "Academic", "Religious", "Other"], key="single_genre")
                            region = st.text_input("Specific region:", key="single_region")
                        
                        metadata = {
                            "source": source,
                            "author": author,
                            "genre": genre,
                            "region": region,
                            "original_filename": uploaded_file.name
                        }
                        
                        if st.button("💾 Save to Corpus"):
                            filename = f"{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            if save_corpus_file(selected_corpus_dialect, filename, content, metadata):
                                st.success(f"✅ File saved to {selected_corpus_dialect} corpus!")
                                st.balloons()
                            else:
                                st.error("❌ Failed to save file.")
                    
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            
            elif upload_method == "Multiple Files":
                uploaded_files = st.file_uploader(
                    "Choose multiple text files", 
                    type=['txt'],
                    accept_multiple_files=True,
                    help="Upload multiple text files at once"
                )
                
                if uploaded_files:
                    st.write(f"Selected {len(uploaded_files)} files")
                    
                    # Bulk metadata
                    st.markdown("**Bulk Metadata (Applied to all files):**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        bulk_source = st.text_input("Source:", key="bulk_source")
                        bulk_genre = st.selectbox("Genre:", ["Literary", "News", "Conversational", "Academic", "Religious", "Other"], key="bulk_genre")
                    with col_b:
                        bulk_author = st.text_input("Author/Speaker:", key="bulk_author")
                        bulk_region = st.text_input("Region:", key="bulk_region")
                    
                    if st.button("💾 Save All to Corpus"):
                        success_count = 0
                        for uploaded_file in uploaded_files:
                            try:
                                content = uploaded_file.read().decode('utf-8-sig')
                                filename = f"{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                
                                metadata = {
                                    "source": bulk_source,
                                    "author": bulk_author,
                                    "genre": bulk_genre,
                                    "region": bulk_region,
                                    "original_filename": uploaded_file.name
                                }
                                
                                if save_corpus_file(selected_corpus_dialect, filename, content, metadata):
                                    success_count += 1
                            except Exception as e:
                                st.error(f"Error with file {uploaded_file.name}: {e}")
                        
                        if success_count > 0:
                            st.success(f"✅ Successfully saved {success_count}/{len(uploaded_files)} files!")
                            st.balloons()
            
            elif upload_method == "Text Input":
                # Direct text input
                direct_text = st.text_area(
                    "Enter text directly:", 
                    height=200,
                    placeholder="یہاں اردو متن داخل کریں..."
                )
                
                if direct_text.strip():
                    # Metadata for direct input
                    st.markdown("**Text Metadata:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        direct_source = st.text_input("Source:", key="direct_source")
                        direct_author = st.text_input("Author/Speaker:", key="direct_author")
                    with col_b:
                        direct_genre = st.selectbox("Genre:", ["Literary", "News", "Conversational", "Academic", "Religious", "Other"], key="direct_genre")
                        direct_region = st.text_input("Region:", key="direct_region")
                    
                    if st.button("💾 Save Text to Corpus"):
                        filename = f"direct_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        metadata = {
                            "source": direct_source,
                            "author": direct_author,
                            "genre": direct_genre,
                            "region": direct_region,
                            "input_method": "direct"
                        }
                        
                        if save_corpus_file(selected_corpus_dialect, filename, direct_text, metadata):
                            st.success("✅ Text saved to corpus!")
                            st.balloons()
        
        with col2:
            st.markdown("### 📊 Current Corpus Status")
            stats = get_corpus_statistics()
            
            if stats:
                for dialect, stat in stats.items():
                    with st.container():
                        st.markdown(f"""
                        **{dialect}**
                        - Files: {stat['total_texts']}
                        - Words: {stat['total_words']:,}
                        - Characters: {stat['total_chars']:,}
                        """)
                        st.markdown("---")
            else:
                st.info("No corpus data available yet.")
    
    with corpus_tab2:
        st.subheader("📋 Browse Corpus Files")
        
        # Dialect filter for browsing
        browse_dialect = st.selectbox("Select dialect to browse:", ["All"] + available_dialects, key="browse_dialect")
        
        corpus_stats = get_corpus_statistics()
        
        if browse_dialect == "All":
            dialects_to_show = corpus_stats.keys()
        else:
            dialects_to_show = [browse_dialect] if browse_dialect in corpus_stats else []
        
        for dialect in dialects_to_show:
            if dialect in corpus_stats:
                st.markdown(f"### {dialect}")
                
                files_info = []
                index_data = load_corpus_index()
                
                if dialect in index_data:
                    for filename, file_info in index_data[dialect].items():
                        files_info.append({
                            "Filename": filename,
                            "Upload Date": file_info.get("upload_date", "Unknown"),
                            "Words": file_info.get("word_count", 0),
                            "Characters": file_info.get("char_count", 0),
                            "Source": file_info.get("metadata", {}).get("source", "Unknown"),
                            "Genre": file_info.get("metadata", {}).get("genre", "Unknown")
                        })
                    
                    if files_info:
                        files_df = pd.DataFrame(files_info)
                        st.dataframe(files_df, use_container_width=True)
                        
                        # File viewer
                        selected_file = st.selectbox(f"View file from {dialect}:", ["Select a file..."] + [f["Filename"] for f in files_info], key=f"view_{dialect}")
                        
                        if selected_file != "Select a file...":
                            content = load_corpus_file(dialect, selected_file)
                            if content:
                                st.text_area(f"Content of {selected_file}:", content, height=300, key=f"content_{dialect}_{selected_file}")
                    else:
                        st.info(f"No files found for {dialect}")
                
                st.markdown("---")
    
    with corpus_tab3:
        st.subheader("🔍 Search Corpus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Enter search term:", placeholder="Search in corpus...")
            search_dialect = st.selectbox("Search in dialect:", ["All Dialects"] + available_dialects, key="search_dialect")
            case_sensitive = st.checkbox("Case sensitive search")
            
            if search_query:
                search_dialect_param = None if search_dialect == "All Dialects" else search_dialect
                results = search_corpus(search_dialect_param, search_query, case_sensitive)
                
                if results:
                    st.markdown(f"### Found {len(results)} files with '{search_query}'")
                    
                    for result in results:
                        with st.expander(f"{result['dialect']} - {result['filename']} ({result['matches']} matches)"):
                            st.markdown(f"**Dialect:** {result['dialect']}")
                            st.markdown(f"**File:** {result['filename']}")
                            st.markdown(f"**Total matches:** {result['matches']}")
                            
                            st.markdown("**Context snippets:**")
                            for i, context in enumerate(result['contexts']):
                                # Highlight the search term in context
                                highlighted = context['context'].replace(
                                    search_query, 
                                    f"**{search_query}**"
                                )
                                st.markdown(f"*{i+1}.* ...{highlighted}...")
                else:
                    st.info(f"No results found for '{search_query}'")
        
        with col2:
            st.markdown("### 🔧 Search Tips")
            st.markdown("""
            - Use Urdu script for better results
            - Search is performed across all text content
            - Case sensitivity can be toggled
            - Results show context around matches
            """)
    
    with corpus_tab4:
        st.subheader("📈 Corpus Analytics")
        
        analytics_dialect = st.selectbox("Select dialect for analysis:", available_dialects, key="analytics_dialect")
        
        if st.button("🔄 Generate Analysis"):
            with st.spinner("Analyzing corpus..."):
                analysis = analyze_corpus_linguistics(analytics_dialect)
                
                if analysis:
                    # Display analytics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Files", analysis['file_count'])
                    with col2:
                        st.metric("Total Words", f"{analysis['total_words']:,}")
                    with col3:
                        st.metric("Unique Words", f"{analysis['unique_words']:,}")
                    with col4:
                        st.metric("Lexical Diversity", analysis['lexical_diversity'])
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Language Statistics")
                        st.metric("Average Word Length", f"{analysis['avg_word_length']} characters")
                        st.metric("Average Sentence Length", f"{analysis['avg_sentence_length']} words")
                        st.metric("Total Characters", f"{analysis['total_characters']:,}")
                    
                    with col2:
                        st.markdown("#### 🔤 Most Common Words")
                        word_freq_df = pd.DataFrame(
                            analysis['most_common_words'], 
                            columns=["Word", "Frequency"]
                        )
                        st.dataframe(word_freq_df, use_container_width=True)
                        
                        # Bar chart of top words
                        st.bar_chart(word_freq_df.set_index("Word")["Frequency"])
                
                else:
                    st.warning(f"No corpus data found for {analytics_dialect}")

elif main_tab == "🎤 Audio Input":
    st.markdown("---")
    st.header("🎤 Audio Input & Speech Recognition")
    
    st.markdown("""
    This section allows you to:
    - 📁 Upload audio files in various formats (WAV, MP3, M4A, etc.)
    - 🎙️ Record audio directly in the browser
    - 🔄 Automatically convert speech to text
    - 📚 Add transcribed text to dialect corpus
    """)
    
    # Audio input tabs
    audio_tab1, audio_tab2, audio_tab3 = st.tabs([
        "📁 Upload Audio", 
        "🎙️ Record Audio", 
        "📋 Audio History"
    ])
    
    with audio_tab1:
        st.subheader("📁 Upload Audio File")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Audio file upload
            uploaded_audio = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'],
                help="Upload audio file for speech-to-text conversion"
            )
            
            if uploaded_audio is not None:
                # Display audio player
                st.audio(uploaded_audio, format='audio/wav')
                
                # Audio processing options
                st.markdown("### 🔧 Processing Options")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    # Language selection for recognition
                    recognition_language = st.selectbox(
                        "Recognition Language:",
                        [
                            ("Urdu", "ur-PK"),
                            ("English", "en-US"),
                            ("Hindi", "hi-IN"),
                            ("Arabic", "ar"),
                            ("Auto-detect", "auto")
                        ],
                        format_func=lambda x: x[0]
                    )
                    
                    # Processing method
                    processing_method = st.radio(
                        "Processing Method:",
                        ["Simple (< 1 minute)", "Chunked (longer files)"],
                        help="Choose based on audio length"
                    )
                
                with col_b:
                    # Dialect assignment
                    available_dialects = [
                        "Sindhi-Urdu", "Punjabi-Urdu", "Seraiki-Urdu", 
                        "Pashto-Urdu", "Balochi-Urdu", "Standard Urdu",
                        "Lahori Urdu", "Karachi Urdu", "Peshawari Urdu", "Quetta Urdu"
                    ]
                    audio_dialect = st.selectbox("Assign to Dialect:", available_dialects)
                    
                    # Auto-predict dialect from transcript
                    auto_predict_dialect = st.checkbox("🤖 Auto-predict dialect from transcript")
                
                # Transcription button
                if st.button("🔄 Transcribe Audio", type="primary"):
                    with st.spinner("Converting audio and transcribing..."):
                        try:
                            # Convert uploaded file to WAV format
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                # Save uploaded file temporarily
                                tmp_file.write(uploaded_audio.read())
                                uploaded_audio.seek(0)  # Reset file pointer
                                
                                # Convert to WAV
                                wav_file = convert_audio_to_wav(tmp_file.name)
                                
                                if wav_file:
                                    # Transcribe based on selected method
                                    lang_code = recognition_language[1] if recognition_language[1] != "auto" else "ur-PK"
                                    
                                    if processing_method == "Simple (< 1 minute)":
                                        transcript = transcribe_audio_simple(wav_file, lang_code)
                                    else:
                                        transcript = transcribe_audio_chunks(wav_file, lang_code)
                                    
                                    if transcript and transcript.strip():
                                        st.success("✅ Transcription completed!")
                                        
                                        # Auto-predict dialect if enabled
                                        if auto_predict_dialect:
                                            predicted_dialect = predict_dialect(transcript)
                                            st.info(f"🤖 Predicted dialect: **{predicted_dialect}**")
                                            final_dialect = predicted_dialect
                                        else:
                                            final_dialect = audio_dialect
                                        
                                        # Display transcript
                                        st.markdown("### 📝 Transcription Result")
                                        transcribed_text = st.text_area(
                                            "Edit transcription if needed:",
                                            value=transcript,
                                            height=150,
                                            key="audio_transcript"
                                        )
                                        
                                        # Language detection
                                        detected_lang = detect_language_from_audio_text(transcribed_text)
                                        st.info(f"Detected script: {detected_lang.title()}")
                                        
                                        # Metadata for audio
                                        st.markdown("### 📋 Audio Metadata")
                                        col_meta1, col_meta2 = st.columns(2)
                                        
                                        with col_meta1:
                                            speaker_info = st.text_input("Speaker Info:", key="audio_speaker")
                                            recording_location = st.text_input("Recording Location:", key="audio_location")
                                        
                                        with col_meta2:
                                            audio_quality = st.selectbox("Audio Quality:", ["Excellent", "Good", "Fair", "Poor"], key="audio_quality")
                                            audio_context = st.selectbox("Context:", ["Interview", "Conversation", "Monologue", "Reading", "Other"], key="audio_context")
                                        
                                        # Save options
                                        col_save1, col_save2 = st.columns(2)
                                        
                                        with col_save1:
                                            save_audio_file_option = st.checkbox("💾 Save original audio file", value=True)
                                        
                                        with col_save2:
                                            add_to_corpus_audio = st.checkbox("📚 Add transcript to corpus", value=True)
                                        
                                        # Submit button
                                        if st.button("💾 Save Audio & Transcript", key="save_audio_transcript"):
                                            # Prepare metadata
                                            audio_metadata = {
                                                "original_filename": uploaded_audio.name,
                                                "file_type": "audio_upload",
                                                "speaker": speaker_info,
                                                "location": recording_location,
                                                "quality": audio_quality,
                                                "context": audio_context,
                                                "recognition_language": recognition_language[0],
                                                "processing_method": processing_method,
                                                "detected_script": detected_lang,
                                                "transcription_date": datetime.now().isoformat()
                                            }
                                            
                                            success_count = 0
                                            
                                            # Save audio file if requested
                                            if save_audio_file_option:
                                                audio_path = save_audio_file(
                                                    uploaded_audio.read(),
                                                    final_dialect,
                                                    f"upload_{uploaded_audio.name.split('.')[0]}"
                                                )
                                                uploaded_audio.seek(0)  # Reset pointer
                                                
                                                if audio_path:
                                                    audio_metadata["saved_audio_path"] = audio_path
                                                    success_count += 1
                                            
                                            # Save transcript to corpus if requested
                                            if add_to_corpus_audio:
                                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                                transcript_filename = f"audio_transcript_{timestamp}.txt"
                                                
                                                if save_corpus_file(final_dialect, transcript_filename, transcribed_text, audio_metadata):
                                                    success_count += 1
                                            
                                            if success_count > 0:
                                                st.success(f"✅ Successfully saved audio data to {final_dialect} corpus!")
                                                st.balloons()
                                            else:
                                                st.error("❌ Failed to save audio data.")
                                    
                                    else:
                                        st.error("❌ Could not transcribe audio. Please try a different file or processing method.")
                                    
                                    # Clean up temporary files
                                    try:
                                        os.unlink(wav_file)
                                        os.unlink(tmp_file.name)
                                    except:
                                        pass
                                
                                else:
                                    st.error("❌ Could not process audio file.")
                        
                        except Exception as e:
                            st.error(f"❌ Error processing audio: {e}")
        
        with col2:
            st.markdown("### 📊 Audio Processing Info")
            st.markdown("""
            **Supported Formats:**
            - WAV, MP3, M4A
            - OGG, FLAC, AAC
            
            **Recognition Languages:**
            - Urdu (Primary)
            - English (Fallback)
            - Hindi, Arabic
            
            **Processing Tips:**
            - Clear audio works best
            - Avoid background noise
            - Speak clearly and slowly
            - Use "Chunked" for files > 1 minute
            """)
    
    with audio_tab2:
        st.subheader("🎙️ Record Audio Directly")
        
        st.markdown("**Note:** This feature requires browser microphone permissions.")
        
        # Audio recorder component
        audio_bytes = st_audiorec()
        
        if audio_bytes is not None:
            st.audio(audio_bytes, format='audio/wav')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Recording settings
                st.markdown("### 🔧 Recording Settings")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    record_dialect = st.selectbox(
                        "Assign to Dialect:", 
                        available_dialects, 
                        key="record_dialect"
                    )
                    record_language = st.selectbox(
                        "Recognition Language:",
                        [("Urdu", "ur-PK"), ("English", "en-US"), ("Auto-detect", "auto")],
                        format_func=lambda x: x[0],
                        key="record_language"
                    )
                
                with col_b:
                    auto_predict_record = st.checkbox("🤖 Auto-predict dialect", key="auto_predict_record")
                    immediate_transcribe = st.checkbox("🔄 Auto-transcribe", value=True, key="immediate_transcribe")
                
                if immediate_transcribe or st.button("🔄 Transcribe Recording"):
                    with st.spinner("Transcribing recorded audio..."):
                        try:
                            # Save recorded audio to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(audio_bytes)
                                
                                # Transcribe
                                lang_code = record_language[1] if record_language[1] != "auto" else "ur-PK"
                                transcript = transcribe_audio_simple(tmp_file.name, lang_code)
                                
                                if transcript and transcript.strip():
                                    st.success("✅ Recording transcribed!")
                                    
                                    # Auto-predict dialect
                                    if auto_predict_record:
                                        predicted_dialect = predict_dialect(transcript)
                                        st.info(f"🤖 Predicted dialect: **{predicted_dialect}**")
                                        final_record_dialect = predicted_dialect
                                    else:
                                        final_record_dialect = record_dialect
                                    
                                    # Show transcript
                                    st.markdown("### 📝 Recording Transcript")
                                    edited_transcript = st.text_area(
                                        "Edit if needed:",
                                        value=transcript,
                                        height=100,
                                        key="record_transcript"
                                    )
                                    
                                    # Quick metadata
                                    record_context = st.selectbox(
                                        "Context:", 
                                        ["Practice", "Interview", "Conversation", "Reading", "Other"],
                                        key="record_context"
                                    )
                                    
                                    # Save recording
                                    if st.button("💾 Save Recording & Transcript", key="save_recording"):
                                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        
                                        record_metadata = {
                                            "file_type": "browser_recording",
                                            "context": record_context,
                                            "recognition_language": record_language[0],
                                            "recording_date": datetime.now().isoformat(),
                                            "predicted_dialect": final_record_dialect if auto_predict_record else None
                                        }
                                        
                                        # Save audio
                                        audio_path = save_audio_file(
                                            audio_bytes,
                                            final_record_dialect,
                                            f"recording_{timestamp}"
                                        )
                                        
                                        # Save transcript
                                        transcript_filename = f"recording_transcript_{timestamp}.txt"
                                        
                                        if audio_path and save_corpus_file(final_record_dialect, transcript_filename, edited_transcript, record_metadata):
                                            st.success(f"✅ Recording saved to {final_record_dialect} corpus!")
                                            st.balloons()
                                        else:
                                            st.error("❌ Failed to save recording.")
                                
                                else:
                                    st.error("❌ Could not transcribe recording. Please try again.")
                                
                                # Clean up
                                try:
                                    os.unlink(tmp_file.name)
                                except:
                                    pass
                        
                        except Exception as e:
                            st.error(f"❌ Error processing recording: {e}")
            
            with col2:
                st.markdown("### 🎙️ Recording Tips")
                st.markdown("""
                **For Best Results:**
                - Speak clearly
                - Avoid background noise
                - Hold microphone steady
                - Speak at normal pace
                - Keep recordings under 30 seconds for quick processing
                """)
    
    with audio_tab3:
        st.subheader("📋 Audio Transcription History")
        
        # Display audio files from corpus
        corpus_stats = get_corpus_statistics()
        
        if corpus_stats:
            for dialect in corpus_stats.keys():
                audio_dir = Path(f"corpus/{dialect.replace(' ', '_').replace('-', '_')}/audio")
                
                if audio_dir.exists():
                    audio_files = list(audio_dir.glob("*.wav"))
                    
                    if audio_files:
                        st.markdown(f"### {dialect} Audio Files")
                        
                        for audio_file in audio_files:
                            with st.expander(f"🎵 {audio_file.name}"):
                                # Display audio player
                                try:
                                    with open(audio_file, "rb") as f:
                                        audio_data = f.read()
                                    st.audio(audio_data, format='audio/wav')
                                    
                                    # Show file info
                                    file_stat = audio_file.stat()
                                    st.markdown(f"**File size:** {file_stat.st_size / 1024:.1f} KB")
                                    st.markdown(f"**Modified:** {datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                                    
                                    # Find corresponding transcript
                                    transcript_name = audio_file.stem.replace("audio_", "").replace("upload_", "").replace("recording_", "")
                                    
                                    # Look for transcript in corpus files
                                    index_data = load_corpus_index()
                                    transcript_found = False
                                    
                                    if dialect in index_data:
                                        for filename, file_info in index_data[dialect].items():
                                            if transcript_name in filename and "transcript" in filename:
                                                transcript_content = load_corpus_file(dialect, filename)
                                                if transcript_content:
                                                    st.markdown("**Transcript:**")
                                                    st.text_area("", value=transcript_content, height=100, key=f"hist_{audio_file.name}")
                                                    transcript_found = True
                                                    break
                                    
                                    if not transcript_found:
                                        st.info("No transcript found for this audio file.")
                                
                                except Exception as e:
                                    st.error(f"Error displaying audio: {e}")
        else:
            st.info("No audio files found in corpus.")

elif main_tab == "💬 Text Input":
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
