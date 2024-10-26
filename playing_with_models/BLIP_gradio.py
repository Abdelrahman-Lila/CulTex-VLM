import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

interface = gr.Interface(
    fn=generate_caption,                  # Function to call
    inputs=gr.Image(type="pil"),    # Input: Image (PIL format)
    outputs=gr.Textbox(),          # Output: Text (caption)
    live=True                               # Optional: For real-time feedback
)

interface.launch(share=True)
