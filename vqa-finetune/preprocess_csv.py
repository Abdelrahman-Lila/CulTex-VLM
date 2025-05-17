import os
import re
import requests
import pandas as pd
from tqdm import tqdm

input_csv = "unique_questions_reduced.csv"               
output_csv = "data_unique_preprocessed.csv"  
image_cache_dir = "image_cache_output"      
alpha = 1

os.makedirs(image_cache_dir, exist_ok=True)

def download_image(link, cache_dir):
    """
    Downloads the image from a Google Drive link.
    Extracts the file ID from the URL and saves the image as <file_id>.png in cache_dir.
    Returns the local file path.
    """
    match = re.search(r"/d/([^/]+)", link)
    if not match:
        raise ValueError(f"Invalid Google Drive URL format: {link}")
    file_id = match.group(1)
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    local_filename = os.path.join(cache_dir, f"{file_id}.png")
    if not os.path.exists(local_filename):
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(local_filename, "wb") as f:
                f.write(response.content)
        else:
            print(f"Warning: Failed to download image from {link} (status code {response.status_code}).")
            return None
    return local_filename

def main():
    df = pd.read_csv(input_csv)
    
    #  small seeded fraction
    if alpha < 1.0:
        subset_size = int(len(df) * alpha)
        df = df.iloc[:subset_size].reset_index(drop=True)
    
    # for random sampling:
    # if alpha < 1.0:
    #     df = df.sample(frac=alpha, random_state=42).reset_index(drop=True)
    
    local_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        link = row["Link"]
        try:
            local_path = download_image(link, image_cache_dir)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            local_path = None
        local_paths.append(local_path)
    
    df["img_path"] = local_paths
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()
