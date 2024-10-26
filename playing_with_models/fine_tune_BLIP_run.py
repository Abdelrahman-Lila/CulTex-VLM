import os
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import BlipProcessor
import json
import torch
from evaluate import load
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import evaluate
excel_file = '/home/abdelrahman.elsayed/GP/dataset/data set.xlsx'  
image_root_folder = '/home/abdelrahman.elsayed/GP/dataset/training_set'  
df = pd.read_excel(excel_file)
# Load BLEU metric
bleu = evaluate.load("bleu")
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_image_caption_pairs(df, image_root_folder):
    data = []
    
    for idx, row in df.iterrows():
        subfolder_name = row['exhibits']  
        caption_english = row['Text in English']

        subfolder_path = os.path.join(image_root_folder, f"{idx+1}_{subfolder_name}")

        if os.path.exists(subfolder_path):
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)

                data.append({
                    "image_path": image_path,
                    "caption_english": caption_english
                })
        else:
            print(f"Warning: Subfolder {subfolder_path} does not exist!")

    return data

data = create_image_caption_pairs(df, image_root_folder)
dataset = Dataset.from_list(data)

def process_data(examples):
    images = [Image.open(image_path) for image_path in examples["image_path"]]
    
    caption_english = examples['caption_english']

    inputs = processor(images=images, text=caption_english, padding=True,truncation=True,  return_tensors="pt").to(device, torch.float16)
    
    inputs["labels"] = inputs["input_ids"]
    return inputs
dataset = dataset.map(process_data, batched=True , batch_size=1)
# Load BLIP model and processor
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("/home/abdelrahman.elsayed/GP/playing_with_models/BLIP_model")
def generate_caption(image_path, max_new_tokens=300): 
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print("Train Data set" , train_dataset)
print("EVAL data set" , eval_dataset)
eval_dataset = eval_dataset.map(lambda example: {'generated_caption': generate_caption(example['image_path'])} , batched=True , batch_size=1)
print("Mapping finished")
print(eval_dataset)
output_json_file = 'image_caption_data.json'
dataset.to_json('dataset_output.json', batch_size=16)

# print(f"Data successfully dumped to {output_json_file}")

def fine_tune_save(dataset , model , processor):
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print("Train Data set" , train_dataset)
    print("EVAL data set" , eval_dataset)
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        evaluation_strategy="epoch",     # Evaluate every epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        weight_decay=0.01,
        save_total_limit=2,
        remove_unused_columns=False,  
        logging_dir='./logs',           
    )

    trainer = Trainer(
        model=model,                 
        args=training_args,          
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,    
        tokenizer=processor,          
        data_collator=None,          
    )

    trainer.train()


    model_save_path = "/home/abdelrahman.elsayed/GP/playing_with_models/BLIP_model"
    processor_save_path = "/home/abdelrahman.elsayed/GP/playing_with_models/BLIP_processor"

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(processor_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Processor saved to {processor_save_path}")


# def benchmarkModel(model, dataset, processor):
#     predictions = dataset['generated_caption']
#     print(predictions)
    
#     references = [[ref.split()] for ref in dataset['caption_english']] 
#     print(references)
#     # Compute BLEU score
#     bleu_result = bleu.compute(predictions=predictions, references=references)
    
#     print(f"BLEU Score: {bleu_result['bleu']}")


def benchmarkModel(model, dataset, processor):
    # Load the ROUGE metric
    rouge = load("rouge")
    
    # Get predictions from the dataset
    predictions = dataset['generated_caption']
    print("Predictions:", predictions)
    
    # Prepare references for ROUGE
    references = [ref.split() for ref in dataset['caption_english']]
    print("References:", references)
    
    # Compute ROUGE score
    rouge_result = rouge.compute(predictions=predictions, references=references)
    
    # Print ROUGE results
    print(f"ROUGE-1 Score: {rouge_result['rouge1']}")
    print(f"ROUGE-2 Score: {rouge_result['rouge2']}")
    print(f"ROUGE-L Score: {rouge_result['rougeL']}")

def main():
    # benchmarkModel(model , eval_dataset , processor)
    print("HERE")
if __name__ == "__main__":
    main()

