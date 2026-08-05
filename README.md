# Spotify Tracks Classification and Recommendation System

This repository contains the final project for the Data Science and Machine Learning certification. The project uses a Spotify dataset of 114,000 tracks spanning 114 genres to perform exploratory data analysis, train a genre classification model, and implement a content-based recommendation system. A Streamlit web interface is provided to interact with the models.

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
