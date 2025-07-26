import os
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import chromadb

def load_dataset(base_path, excel_path):
    # Load Excel data
    df = pd.read_excel(excel_path)
    data = []
    
    # Create exhibit to caption mapping
    caption_map = {row['exhibits']: row['Text in English'] for _, row in df.iterrows()}
    
    # Process folders
    for folder in sorted(os.listdir(base_path)):
        if not os.path.isdir(os.path.join(base_path, folder)):
            continue
            
        # Extract exhibit name (remove numeric prefix)
        exhibit_name = "_".join(folder.split("_")[1:])
        
        # Get all images for this exhibit
        image_files = [f for f in os.listdir(os.path.join(base_path, folder)) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create entries
        for img_file in image_files:
            data.append({
                "exhibit": exhibit_name,
                "caption": caption_map.get(exhibit_name, ""),
                "image_path": os.path.join(base_path, folder, img_file)
            })
    
    return pd.DataFrame(data)

train_df = load_dataset("/home/abdelrahman.elsayed/GP/dataset/training_set", "/home/abdelrahman.elsayed/GP/dataset/data set.xlsx")
test_df = load_dataset("/home/abdelrahman.elsayed/GP/dataset/test_set", "/home/abdelrahman.elsayed/GP/dataset/data set.xlsx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_embeddings(df):
    embeddings = []
    for _, row in df.iterrows():
        try:
            image = Image.open(row['image_path'])
            inputs = processor(
                text=None,  # Image-only embedding
                images=image,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                
            embeddings.append({
                "exhibit": row['exhibit'],
                "caption": row['caption'],
                "embedding": features.cpu().numpy()[0]
            })
        except Exception as e:
            print(f"Error processing {row['image_path']}: {str(e)}")
    
    return embeddings

train_embeddings = generate_embeddings(train_df)

client = chromadb.Client()
collection = client.create_collection("exhibits")

# Add embeddings
collection.add(
    ids=[str(i) for i in range(len(train_embeddings))],
    embeddings=[e['embedding'].tolist() for e in train_embeddings],
    metadatas=[{"caption": e['caption']} for e in train_embeddings]
)

def retrieve_with_chroma(query_emb):
    return collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=3
    )

