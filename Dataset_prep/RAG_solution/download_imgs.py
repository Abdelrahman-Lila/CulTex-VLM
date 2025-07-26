import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Paths
csv_file = "/media/gamil/Windows-SSD/FOLDER/STUDY/EJUST/Graduation Project/Dataset_prep/RAG_solution/dataset_photos.csv"
output_folder = "/media/gamil/Windows-SSD/FOLDER/STUDY/EJUST/Graduation Project/Dataset_prep/RAG_solution/ImagesDB"
df = pd.read_csv(csv_file)
df["Index"] = (df["Index"].astype(int) - 72).astype(str)
print(df.head())

os.makedirs(output_folder, exist_ok=True)

# Ensure the column exists
if "downloaded_photo_path" not in df.columns:
    df["downloaded_photo_path"] = None


# Downloader
def download_image(url, save_path):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


# Download loop
for index, row in df.iterrows():
    # If already has path and file exists, skip
    existing_path = row.get("downloaded_photo_path")
    if existing_path and os.path.exists(existing_path):
        print(f"Skipping {index}, already downloaded.")
        continue

    photo_url = row["photo"]
    ext = os.path.splitext(urlparse(photo_url).path)[1] or ".jpg"
    filename = f"img_{index}{ext}"
    save_path = os.path.join(output_folder, filename)

    downloaded_path = download_image(photo_url, save_path)
    df.at[index, "downloaded_photo_path"] = downloaded_path

# Save updated CSV
df.to_csv("updated_file.csv", index=False)
print("Updated CSV saved to updated_file.csv")
