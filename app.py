import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Set Page Config
st.set_page_config(
    page_title="Spotify Insights & Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Spotify Theme Styling
st.markdown("""
<style>
    /* Dark Mode Global Background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Modify sidebar style */
    [data-testid="stSidebar"] {
        background-color: #191414;
        color: #ffffff;
    }
    
    /* Streamlit components hover state */
    .stButton>button {
        background-color: #1DB954 !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 2rem !important;
        transition: transform 0.2s, background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.05);
    }
    
    /* Metrics box */
    [data-testid="stMetricValue"] {
        color: #1DB954;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    /* Cards for recommendations */
    .spotify-card {
        background-color: #181818;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1DB954;
        transition: transform 0.2s, background-color 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .spotify-card:hover {
        transform: translateY(-5px);
        background-color: #282828;
    }
    .spotify-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .spotify-artist {
        font-size: 0.95rem;
        color: #b3b3b3;
        margin-bottom: 0.5rem;
    }
    .spotify-album {
        font-size: 0.85rem;
        color: #1DB954;
        font-style: italic;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 9px;
        font-size: 1.6rem;
        font-weight: bold;
        color: #b3b3b3;
    }
    .stTabs [aria-selected="true"] {
        color: #1DB954 !important;
        border-bottom-color: #1DB954 !important;
    }

    /* Fix text visibility issues & increase font size */
    label, p, span, .stSlider label, .stCheckbox label, .stRadio label, div[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    
    /* Headings styling */
    h1, .stMarkdown h1 {
        color: #ffffff !important;
        font-size: 3.8rem !important;
        font-weight: bold !important;
    }
    h2, .stMarkdown h2 {
        color: #ffffff !important;
        font-size: 3.0rem !important;
        font-weight: bold !important;
    }
    h3, .stMarkdown h3 {
        color: #ffffff !important;
        font-size: 2.4rem !important;
        font-weight: bold !important;
    }
    h4, .stMarkdown h4 {
        color: #ffffff !important;
        font-size: 1.9rem !important;
        font-weight: bold !important;
    }
    
    /* Selectbox titles and drop downs */
    .stSelectbox label p, .stTextInput label p, .stNumberInput label p {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    /* Subtitles / auxiliary text visibility */
    .stMarkdown div p {
        color: #e0e0e0 !important;
        font-size: 1.2rem !important;
    }
    
    /* Keep metric value green and increase size */
    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
        font-size: 2.6rem !important;
    }

    /* Metric labels should be clear light grey */
    [data-testid="stMetricLabel"] p {
        color: #cccccc !important;
        font-size: 1.1rem !important;
        font-weight: normal !important;
    }

    /* Force all Streamlit UI SVG icons to be white (excluding Plotly charts) */
    [data-testid="stIcon"] svg, 
    button svg, 
    [data-baseweb="icon"] svg, 
    [data-baseweb="select"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Modify sliders to be Spotify Green */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #1DB954 !important;
    }
    
    /* Styling input fields, text areas, number boxes (removing selectbox from background override to preserve dropdown arrow) */
    .stTextInput input, .stNumberInput input {
        font-size: 1.4rem !important;
        color: #ffffff !important;
        background-color: #181818 !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        font-size: 1.4rem !important;
    }
    /* Enlarge slider output values */
    div[data-testid="stSlider"] div {
        font-size: 1.35rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Define Paths
PROJECT_DIR = r'C:\Users\Public\data science\GUVI PROJECTS\FINAL PROJECT'
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'spotify_tracks.parquet')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset parquet file not found at {DATA_PATH}. Please run the setup scripts first.")
        return None
    df = pd.read_parquet(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

# Load Model
@st.cache_resource
def load_models():
    classifier_path = os.path.join(MODELS_DIR, 'classifier.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    encoder_path = os.path.join(MODELS_DIR, 'encoder.pkl')
    
    if not (os.path.exists(classifier_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        return None, None, None
        
    model = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

df = load_data()
model, scaler, encoder = load_models()

# Sidebar Header
st.sidebar.markdown("<h2 style='text-align: center; color: #1DB954;'>🎵 Spotify Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if df is None:
    st.info("Please verify data directory setup.")
else:
    # Main Header
    st.markdown("<h1 style='color: #ffffff;'><span style='color: #1DB954;'>Spotify</span> Audio Insights & Recommendation System</h1>", unsafe_allow_html=True)
    
    # Check if models trained
    if model is None:
        st.warning("⚠️ Machine Learning models are not detected. Streamlit tabs will have limited classifier functionality until training completes.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎵 Dataset Exploration", "🔮 Genre Classifier", "💿 Smart Recommendations"])
    
    # -------------------- TAB 1: DATASET EXPLORATION --------------------
    with tab1:
        st.markdown("### 📊 Dataset Overview")
        
        # High level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tracks", f"{len(df):,}")
        col2.metric("Total Genres", f"{df['track_genre'].nunique()}")
        col3.metric("Avg Song Length", f"{df['duration_ms'].mean()/60000:.2f} mins")
        explicit_pct = (df['explicit'].sum() / len(df)) * 100
        col4.metric("Explicit Tracks", f"{explicit_pct:.1f}%")
        
        st.markdown("---")
        
        # Grid of Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Audio Feature Distribution")
            feature_to_plot = st.selectbox(
                "Select feature to view distribution:",
                ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']
            )
            fig_hist = px.histogram(
                df, x=feature_to_plot, nbins=50, 
                color_discrete_sequence=['#1DB954'],
                title=f"Distribution of {feature_to_plot.capitalize()}"
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=15),
                xaxis=dict(gridcolor='#282828', tickfont=dict(color='#ffffff', size=15), titlefont=dict(color='#ffffff', size=17)),
                yaxis=dict(gridcolor='#282828', tickfont=dict(color='#ffffff', size=15), titlefont=dict(color='#ffffff', size=17))
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with chart_col2:
            st.markdown("#### Feature Relationships")
            features_list = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']
            x_feature = st.selectbox("X Axis Feature:", features_list, index=0)
            y_feature = st.selectbox("Y Axis Feature:", features_list, index=6)
            
            # Subsample for faster plotting
            sample_df = df.sample(2000, random_state=42)
            fig_scatter = px.scatter(
                sample_df, x=x_feature, y=y_feature, color='track_genre',
                hover_data=['track_name', 'artists'],
                title=f"{x_feature.capitalize()} vs {y_feature.capitalize()} (Sample of 2000 tracks)"
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=15),
                xaxis=dict(gridcolor='#282828', tickfont=dict(color='#ffffff', size=15), titlefont=dict(color='#ffffff', size=17)),
                yaxis=dict(gridcolor='#282828', tickfont=dict(color='#ffffff', size=15), titlefont=dict(color='#ffffff', size=17)),
                legend=dict(font=dict(size=13, color='#ffffff'))
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("#### 🌡️ Feature Correlation Heatmap")
        numeric_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        corr = df[numeric_cols].corr()
        fig_heat = px.imshow(
            corr,
            labels=dict(x="", y="", color="Correlation"),
            x=numeric_cols,
            y=numeric_cols,
            color_continuous_scale='Greens',
            zmin=-1, zmax=1
        )
        fig_heat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=15),
            width=900,
            height=850,
            margin=dict(l=150, r=150, t=50, b=150),
            xaxis=dict(tickfont=dict(color='#ffffff', size=15)),
            yaxis=dict(tickfont=dict(color='#ffffff', size=15)),
            coloraxis_colorbar=dict(
                title=dict(text="Correlation", font=dict(size=17, color='#ffffff')),
                tickfont=dict(size=14, color='#ffffff'),
                len=0.8
            )
        )
        col_heat_l, col_heat_m, col_heat_r = st.columns([1, 6, 1])
        with col_heat_m:
            st.plotly_chart(fig_heat, use_container_width=True)
        
    # -------------------- TAB 2: GENRE CLASSIFIER --------------------
    with tab2:
        st.markdown("### 🔮 Predict Song Genre")
        st.write("Tune the audio characteristics below to see what music genre the model predicts.")
        
        if model is None:
            st.error("Model files not found. Please train the model by running the `train.py` script.")
        else:
            # Inputs
            in_col1, in_col2, in_col3 = st.columns(3)
            
            with in_col1:
                popularity = st.slider("Popularity", 0, 100, 50)
                duration_ms = st.number_input("Duration (ms)", min_value=1000, max_value=3600000, value=200000)
                explicit = st.checkbox("Explicit Lyrics?", value=False)
                danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
                energy = st.slider("Energy", 0.0, 1.0, 0.6)
                
            with in_col2:
                key = st.slider("Key (Standard Pitch Class)", -1, 11, 2)
                loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0)
                mode = st.radio("Mode", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor")
                speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
                acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
                
            with in_col3:
                instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1)
                liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
                valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.5)
                tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0)
                time_signature = st.slider("Time Signature", 3, 7, 4)
                
            if st.button("🔮 Predict Genre"):
                # Prepare input array
                input_data = pd.DataFrame([{
                    'popularity': popularity,
                    'duration_ms': duration_ms,
                    'explicit': int(explicit),
                    'danceability': danceability,
                    'energy': energy,
                    'key': key,
                    'loudness': loudness,
                    'mode': mode,
                    'speechiness': speechiness,
                    'acousticness': acousticness,
                    'instrumentalness': instrumentalness,
                    'liveness': liveness,
                    'valence': valence,
                    'tempo': tempo,
                    'time_signature': time_signature
                }])
                
                # Preprocess & Scale
                input_scaled = scaler.transform(input_data)
                
                # Predict Probabilities
                probs = model.predict_proba(input_scaled)[0]
                
                # Get Top 5 Predictions
                top5_idx = np.argsort(probs)[-5:][::-1]
                top5_genres = encoder.inverse_transform(top5_idx)
                top5_probs = probs[top5_idx] * 100
                
                # Display Results
                st.markdown("---")
                st.markdown("### Top 5 Predicted Genres")
                
                res_fig = go.Figure(go.Bar(
                    x=top5_probs,
                    y=top5_genres,
                    orientation='h',
                    marker=dict(color='#1DB954'),
                    text=[f"{p:.1f}%" for p in top5_probs],
                    textposition='auto',
                ))
                res_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', size=15),
                    xaxis=dict(title="Probability (%)", tickfont=dict(color='#ffffff', size=16), titlefont=dict(color='#ffffff', size=18), gridcolor='#282828'),
                    yaxis=dict(tickfont=dict(color='#ffffff', size=16), autorange="reversed")
                )
                st.plotly_chart(res_fig, use_container_width=True)

    # -------------------- TAB 3: SMART RECOMMENDATIONS --------------------
    with tab3:
        st.markdown("### 💿 Content-Based Track Recommendations")
        st.write("Find songs similar to your favorite track based on their acoustic features.")
        
        # Setup selection list
        # To avoid overloading selectbox, search by artist or song
        search_query = st.text_input("🔍 Search for a song title or artist name in the database:")
        
        if search_query:
            # Filter matches
            matches = df[
                df['track_name'].str.contains(search_query, case=False, na=False) |
                df['artists'].str.contains(search_query, case=False, na=False)
            ].head(100) # limit to 100
            
            if matches.empty:
                st.info("No matching songs found. Try a different term.")
            else:
                # Select dropdown
                song_list = [f"{row['track_name']} — {row['artists']} ({row['album_name']})" for _, row in matches.iterrows()]
                selected_song_str = st.selectbox("Select a song from search results:", song_list)
                
                if selected_song_str:
                    # Get index of selected song
                    selected_idx = matches.index[song_list.index(selected_song_str)]
                    selected_track = df.loc[selected_idx]
                    
                    # Display Selected Track Features
                    st.markdown("#### Selected Track Properties")
                    col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)
                    col_feat1.markdown(f"**Title**: {selected_track['track_name']}")
                    col_feat2.markdown(f"**Artist**: {selected_track['artists']}")
                    col_feat3.markdown(f"**Album**: {selected_track['album_name']}")
                    col_feat4.markdown(f"**Genre**: `{selected_track['track_genre']}`")
                    
                    st.markdown("---")
                    
                    # Fit Nearest Neighbors on the fly
                    # Using features: audio descriptors
                    rec_features = [
                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
                    ]
                    
                    # Scale recommendation features
                    rec_scaler = StandardScaler()
                    scaled_rec_feats = rec_scaler.fit_transform(df[rec_features])
                    
                    # Train KNN
                    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
                    knn.fit(scaled_rec_feats)
                    
                    # Query for selected track
                    query_feat = scaled_rec_feats[selected_idx].reshape(1, -1)
                    distances, indices = knn.kneighbors(query_feat)
                    
                    # Exclude the queried track itself (which is at index 0)
                    indices = indices[0][1:]
                    distances = distances[0][1:]
                    
                    # Display recommendations
                    st.markdown("#### Recommended Songs")
                    rec_df = df.iloc[indices]
                    
                    # Display in grid
                    grid_cols = st.columns(2)
                    for i, (_, row) in enumerate(rec_df.iterrows()):
                        col = grid_cols[i % 2]
                        sim_score = (1 - distances[i]) * 100
                        col.markdown(f"""
                        <div class="spotify-card">
                            <div class="spotify-title">{row['track_name']}</div>
                            <div class="spotify-artist">By {row['artists']}</div>
                            <div class="spotify-album">Album: {row['album_name']}</div>
                            <div style="font-size: 0.85rem; margin-top: 5px; color: #b3b3b3;">
                                Genre: <span style="color: #1DB954; font-weight: bold;">{row['track_genre']}</span> | Similarity: {sim_score:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Type a song title or artist name above (e.g., 'Coldplay', 'Beatles', 'Acoustic') to get recommendations.")
