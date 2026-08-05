import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    project_dir = r'C:\Users\Public\data science\GUVI PROJECTS\FINAL PROJECT'
    data_path = os.path.join(project_dir, 'data', 'spotify_tracks.parquet')
    models_dir = os.path.join(project_dir, 'models')
    
    print(f"Reading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 1. Clean data: Drop index columns or duplicate columns if any
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Drop rows where target is missing (should be 0 anyway)
    df = df.dropna(subset=['track_genre'])
    
    # Convert 'explicit' to numeric (0 or 1)
    df['explicit'] = df['explicit'].astype(int)
    
    # Define features to use for model training
    features = [
        'popularity', 'duration_ms', 'explicit', 'danceability', 
        'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'time_signature'
    ]
    target = 'track_genre'
    
    print(f"Features selected: {features}")
    
    X = df[features].copy()
    y = df[target].copy()
    
    # 2. Encode Target
    print("Encoding target genres...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # 3. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train-Test Split
    print("Splitting into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 5. Train Model
    # ExtraTrees is faster than RandomForest and very effective for this feature set
    print("Training ExtraTreesClassifier model...")
    # Using constraints to keep training time reasonable and avoid memory exhaustion
    # n_jobs=1 is safer on Windows to avoid duplicating memory across multiple spawned processes
    model = ExtraTreesClassifier(
        n_estimators=50, 
        max_depth=12, 
        min_samples_leaf=10,
        random_state=42, 
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # 6. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    # Top-5 Accuracy evaluation (genres can overlap, so top-5 is standard)
    probs = model.predict_proba(X_test)
    top5_correct = 0
    for idx, prob in enumerate(probs):
        top5_preds = np.argsort(prob)[-5:]
        if y_test[idx] in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(y_test)
    print(f"Test Top-5 Accuracy: {top5_acc:.4f}")
    
    # Print classification report for top 10 genres to keep log short
    print("\nClassification Report (Sample of 10 genres):")
    unique_classes = encoder.classes_
    sample_report = classification_report(
        y_test, y_pred, target_names=unique_classes, output_dict=True
    )
    sample_report_df = pd.DataFrame(sample_report).transpose()
    print(sample_report_df.head(10))
    
    # Print Feature Importances
    print("\nFeature Importances:")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print(feat_imp)
    
    # 7. Serialize Models
    print("\nSaving model objects to models/ folder...")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'classifier.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(models_dir, 'encoder.pkl'))
    print("All models serialized successfully!")

if __name__ == "__main__":
    main()
