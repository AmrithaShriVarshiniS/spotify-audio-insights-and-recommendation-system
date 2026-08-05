# Spotify Tracks Classification and Recommendation System

This repository contains the final project for the Data Science and Machine Learning certification. The project uses a Spotify dataset of 114,000 tracks spanning 114 genres to perform exploratory data analysis, train a genre classification model, and implement a content-based recommendation system. A Streamlit web interface is provided to interact with the models.

## Dataset Description

The dataset comprises Spotify tracks spanning across diverse genres. Every track includes specific acoustic features. The details of the columns available in the dataset are listed below:

* **track_id**: Unique Spotify identifier for each track.
* **artists**: Names of artists involved in the track, separated by semicolons for multiple artists.
* **album_name**: Title of the album containing the track.
* **track_name**: Title of the individual track.
* **popularity**: A score from 0 to 100 indicating the track's popularity, primarily based on the track's play count and the recency of plays.
* **duration_ms**: Length of the track in milliseconds.
* **explicit**: Indicates if the track contains explicit lyrics (1 for explicit content; 0 for clean content).
* **danceability**: A metric ranging from 0.0 (least danceable) to 1.0 (most danceable), assessing a track's suitability for dancing based on tempo, rhythm, and beat regularity.
* **energy**: A perceptual measure ranging from 0.0 to 1.0, gauging the track's intensity and activity (e.g., loud, fast, and noisy tracks score close to 1.0).
* **key**: The musical key of the track, represented by integers following standard Pitch Class notation (e.g., 0 = C, 1 = C#, 2 = D). A value of -1 indicates an undetected key.
* **loudness**: Measures the average loudness of the track in decibels (dB).
* **mode**: Indicates the track's modality, with 1 for major mode and 0 for minor mode, determining the type of scale that forms its melodic basis.
* **speechiness**: Assesses the extent of spoken words in a track. Scores above 0.66 indicate tracks composed entirely of spoken words; scores below 0.33 represent music.
* **acousticness**: A scale from 0.0 to 1.0 indicating the likelihood of the track being acoustic, with 1.0 signifying high confidence.
* **instrumentalness**: Estimates the absence of vocals in a track. Values closer to 1.0 suggest a higher probability of the track lacking vocal content.
* **liveness**: Detects the presence of a live audience in the recording (scores above 0.8 strongly indicate a live performance).
* **valence**: A metric ranging from 0.0 to 1.0 describing the track's emotional tone (high valence tracks sound more positive, happy, and cheerful).
* **tempo**: The track's overall estimated tempo measured in beats per minute (BPM).
* **time_signature**: Provides an estimated time signature for the track, indicating the number of beats in each bar (varies from 3 to 7).
* **track_genre**: Specifies the genre classification of the track.

## Project Structure


* **0000 (1).parquet**: The raw dataset containing Spotify tracks and audio characteristics.
* **app.py**: The main Streamlit dashboard file.
* **src/train.py**: Script to preprocess the data and train the classification model.
* **src/move_data.py**: Helper script to initialize directories.
* **models/**: Folder containing the serialized classifier, scaler, and label encoder.
* **data/**: Folder containing the parquet data path.

## Features

### 1. Exploratory Data Analysis
* Displays high-level dataset metrics such as total records, genre counts, and averages.
* Shows histograms of individual audio features (e.g., energy, valence, danceability).
* Computes and visualizes the correlation matrix of numerical features.

### 2. Genre Classification Model
* Accepts manual inputs for acoustic features via sliders.
* Predicts the track genre using an ExtraTreesClassifier.
* Displays the prediction probability distribution for the top 5 most likely genres.

### 3. Content-Based Recommendation System
* Recommends 10 similar tracks for a selected song from the database.
* Uses the K-Nearest Neighbors (KNN) algorithm with Cosine Similarity on standardized audio features.

## Model Training and Metrics

The classification model was trained using scikit-learn. Numerical features were standardized using StandardScaler prior to training.

* **Algorithm**: ExtraTreesClassifier (n_estimators=50, max_depth=12, min_samples_leaf=10)
* **Dataset size**: 114,000 samples, 114 balanced classes
* **Train/Test split**: 80% train, 20% test
* **Test Accuracy**: 25.60%
* **Top-5 Accuracy**: 56.86%

## Setup and Running Instructions

### Dependencies
Install the required Python packages:
```bash
pip install pandas numpy scikit-learn plotly streamlit joblib pyarrow
```

### Run the Web Application
Start the Streamlit application from the project root directory:
```bash
streamlit run app.py
```
Access the application by navigating to `http://localhost:8501` in your browser.
