import os
import shutil

def main():
    src_file = r'C:\Users\admin\.gemini\antigravity\scratch\spotify_tracks.parquet'
    project_dir = r'C:\Users\Public\data science\GUVI PROJECTS\FINAL PROJECT'
    
    print(f"Setting up project directory structure at: {project_dir}")
    
    # Create directories
    dirs = ['data', 'models', 'src']
    for d in dirs:
        path = os.path.join(project_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
        
    # Copy parquet file
    dest_file = os.path.join(project_dir, 'data', 'spotify_tracks.parquet')
    if os.path.exists(src_file):
        print(f"Moving dataset from {src_file} to {dest_file}...")
        shutil.move(src_file, dest_file)
        print("Dataset moved successfully!")
    else:
        if os.path.exists(dest_file):
            print(f"Dataset already exists at destination: {dest_file}")
        else:
            print(f"Error: Source dataset not found at {src_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
