import os
import random
import requests
import wandb
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
from bert_score import score
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from peft import get_peft_model, LoraConfig
import re
import pandas as pd
from nlpaug.augmenter.word import ContextualWordEmbsAug
import yaml
from data import VQADataset

with open("config.yaml" , "r") as f:
        config = yaml.safe_load(f)

from pprint import pprint
pprint(config)
alpha = float(config["model"]["alpha"])       
val_freq = int(config["model"]["val_freq"])        
num_epochs = int(config["model"]["num_epochs"])
patience = int(config["model"]["patience"])
batch_size = int(config["model"]["batch_size"])
learning_rate = float(config["model"]["learning_rate"])
lora_rank = int(config["model"]["lora"]["rank"])
lora_alpha = int(config["model"]["lora"]["alpha"])
lora_dropout = float(config["model"]["lora"]["dropout"])
target_modules = config["model"]["target_modules"]
mode = config["data"]["format"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.manual_seed(42)

exp_name = f"{config['model']['name']}_{config['model']['ft']}_{config['model']['type']}_mode_{mode}"
wandb.init(project="blip-vqa-finetuning", name=exp_name,config={
    "alpha": alpha,
    "val_freq": val_freq,
    "num_epochs": num_epochs,
    "patience": patience,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "target_modules": target_modules
})

def collate_fn(batch):
    # Handle tensor items
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "question": questions,
        "answer": answers,
        "image_path": image_paths
    }

def print_model_parameters_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}\n")

    print("Parameters by module:")
    for name, module in model.named_children():
        total_m = sum(p.numel() for p in module.parameters())
        trainable_m = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen_m = total_m - trainable_m
        print("  " + "-"*50)
        print(f"  {name}:")
        print(f"    Total: {total_m:,}")
        print(f"    Trainable: {trainable_m:,}")
        print(f"    Frozen: {frozen_m:,}")
    print("-"*50, "\n")

if config["model"]["type"] == "base":
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
else:
    print("\nUsing Large Model")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")

print("  " + "-"*50)
print("Initial Stats\n")
print_model_parameters_stats(model)
if config["model"]["ft"] == "lora":
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)

print("Stats After Setting up\n")
print_model_parameters_stats(model)

model.to(device)

if config["data"]["format"] == "csv":
    full_df = pd.read_csv(config["data"]["path"])
    num_total = len(full_df)
    num_train = int(0.9 * num_total)
    num_val = num_total - num_train

    train_df = full_df.sample(n=num_train, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = VQADataset(train_df, processor, is_train=True)
    val_dataset = VQADataset(val_df, processor, is_train=False)

    num_train_subset = int(alpha * len(train_dataset))
    num_val_subset = int(alpha * len(val_dataset))

    if num_train_subset > 0 and num_train_subset < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(num_train_subset))
    if num_val_subset > 0 and num_val_subset < len(val_dataset):
        val_dataset = torch.utils.data.Subset(val_dataset, range(num_val_subset))
else:
    split_p = 0.9
    train_split = f"train[:{int(split_p * 100)}%]"
    valid_split = f"train[{int(split_p * 100)}%:]"

    training_dataset = load_dataset("json", data_files="Data/train.jsonl", split=train_split)
    valid_dataset = load_dataset("json", data_files="Data/train.jsonl", split=valid_split)

    train_dataset = VQADataset(data=training_dataset,
                            processor=processor , is_train=True, data_type="hf")
    val_dataset = VQADataset(data=valid_dataset,
                            processor=processor , is_train=False, data_type="hf")
                            
    num_train_subset = int(alpha * len(train_dataset))
    num_val_subset = int(alpha * len(val_dataset))

    if num_train_subset > 0 and num_train_subset < len(train_dataset):
        train_dataset = torch.utils.data.Subset(train_dataset, range(num_train_subset))
    if num_val_subset > 0 and num_val_subset < len(val_dataset):
        val_dataset = torch.utils.data.Subset(val_dataset, range(num_val_subset))

print("Using subset sizes:")
print(f" - Train: {len(train_dataset)}")
print(f" - Valid: {len(val_dataset)}")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_fn,
    num_workers=2
)
valid_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2
)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate ,  weight_decay=0.01 )
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

min_eval_loss = float("inf")
early_stopping_hook = 0
scaler = torch.cuda.amp.GradScaler()
tracking_information = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1} - Training'):
        input_ids = batch.pop('input_ids').to(device , non_blocking=True)
        pixel_values = batch.pop('pixel_values').to(device , non_blocking=True)
        attention_mask = batch.pop('attention_mask').to(device, non_blocking=True)
        labels = batch.pop('labels').to(device , non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss
        epoch_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    epoch_loss /= len(train_dataloader)
    
    eval_loss = None
    bert_score_f1 = None
    if (epoch + 1) % val_freq == 0 or (epoch == num_epochs - 1):
        model.eval()
        eval_loss = 0.0
        all_generated = []
        all_references = []
        os.makedirs("qualitative", exist_ok=True)
        saved_samples = 0
        # how the model expect the format.
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_dataloader, desc='Validating')):
                input_ids = batch['input_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                answers = batch['answer']
                questions = batch['question']
                image_paths = batch['image_path']
                
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels
                )
                eval_loss += outputs.loss.item()
                
                # Increase max_length and add repetition penalty
                # Modify the generate() call in the validation loop:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_length=50,  # Increased from 30
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=2.5,  # Increased from 2.0
                    temperature=0.7,          # Add temperature for diversity
                    do_sample=True            # Enable sampling-based generation
                )
                generated_answers = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                all_generated.extend(generated_answers)
                all_references.extend(answers)
                
                if batch_idx == 0 and saved_samples < 5:
                    for i in range(len(generated_answers)):
                        if saved_samples >= 10:
                            break
                        img_path = image_paths[i]
                        question = questions[i]
                        gen_answer = generated_answers[i]
                        true_answer = answers[i]
                        
                        # Save image and info
                        img = Image.open(img_path).convert("RGB")
                        exp_dir = f"qualitative/{exp_name}"
                        epoch_dir = os.path.join(f"{exp_dir}", f"epoch_{epoch+1}")
                        os.makedirs(epoch_dir, exist_ok=True)
                        
                        img.save(os.path.join(epoch_dir, f"sample_{saved_samples}.png"))
                        with open(os.path.join(epoch_dir, f"sample_{saved_samples}_info.txt"), "w") as f:
                            f.write(f"Question: {question}\n")
                            f.write(f"Generated Answer: {gen_answer}\n")
                            f.write(f"True Answer: {true_answer}\n")
                        
                        saved_samples += 1

        P, R, F1 = score(all_generated, all_references, lang="en", verbose=False)
        exact_matches = sum(
            [1 if gen.strip().lower() == ref.strip().lower() else 0 
            for gen, ref in zip(all_generated, all_references)]
        )
        exact_match_acc = exact_matches / len(all_references)
        print(f"Exact Matches: {exact_match_acc}")
        bert_score_f1 = F1.mean().item()
        
        eval_loss /= len(valid_dataloader)

        tracking_information.append((
            epoch+1, 
            epoch_loss, 
            eval_loss, 
            bert_score_f1,
            optimizer.param_groups[0]["lr"]
        ))

        if eval_loss < min_eval_loss:
            print(f"Validation improved. Saving model. BERTScore F1: {bert_score_f1:.4f}")
            model.save_pretrained(f"Model/blip-saved-model_{exp_name}", from_pt=True)
            min_eval_loss = eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                print("Early stopping triggered.")
                break

    scheduler.step()

    wandb.log({
    "epoch": epoch+1,
    "train_loss": epoch_loss,
    "eval_loss": eval_loss if eval_loss is not None else None,
    "bert_score_f1": bert_score_f1 if bert_score_f1 is not None else None,
    "learning_rate": optimizer.param_groups[0]["lr"]
})


    print(f"Epoch {epoch+1} / {num_epochs} | Train Loss: {epoch_loss:.4f}", end="")
    if eval_loss is not None:
        print(f" | Eval Loss: {eval_loss:.4f} | BERTScore F1: {bert_score_f1:.4f}", end="")
    print(f" | LR: {optimizer.param_groups[0]['lr']:.6f}")
    

with open("tracking_information.pkl", "wb") as f:
    pickle.dump(tracking_information, f)

wandb.finish() 
print("The finetuning process has finished!")
