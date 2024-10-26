from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch


processor = BlipProcessor.from_pretrained("/home/abdelrahman.elsayed/GP/playing_with_models/BLIP_processor")
model = BlipForConditionalGeneration.from_pretrained("/home/abdelrahman.elsayed/GP/playing_with_models/BLIP_model")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("/home/abdelrahman.elsayed/GP/dataset/test_set/21_A_silo/A_silo_0017.jpg")
prompt = "what is this?"
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs , max_length=500)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
