import pandas as pd
from serpapi import GoogleSearch

API_KEY = "f4eed153e18be835eacdd599506834fdd5710fb9ca34cfb4d3bb946bd13d028e"
CSV_PATH = "/media/ahmed/Work/gp/CultureVLM-DataConstruction/Dataset_CultureVLM_style - Concepts Introductions (Copy).csv"
OUTPUT_PATH = "output_with_images.csv"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Ensure the 'concept' column exists
if "concept" not in df.columns:
    raise ValueError("CSV must contain a 'concept' column")

# Add a new column for image URLs
df["image_url"] = None

# Iterate and fetch image for each concept
for idx, row in df.iterrows():
    concept = row["concept"]
    if pd.isna(concept):
        continue

    query = f"{concept} Egypt"
    params = {"q": query, "tbm": "isch", "api_key": API_KEY}

    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get("images_results", [])

    if images:
        df.at[idx, "image_url"] = images[0]["original"]
    else:
        df.at[idx, "image_url"] = "No image found"

# Save the updated DataFrame
df.to_csv(OUTPUT_PATH, index=False)
print(f"Image URLs saved to {OUTPUT_PATH}")
