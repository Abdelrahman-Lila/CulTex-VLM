import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from nlpaug.augmenter.word import ContextualWordEmbsAug

class VQACSVDataset(Dataset):
    def __init__(self, data, processor, is_train=False):
        self.data = data
        self.processor = processor
        self.is_train = is_train
        if self.is_train:
            self.aug = ContextualWordEmbsAug(
                model_path='bert-base-uncased', 
                action="substitute",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row["Question"]
        answer = row["Answer"]
        image_path = row["img_path"]
        image = Image.open(image_path).convert("RGB")
        
        if self.is_train and random.random() < 0.5:
            try:
                question = self.aug.augment(question)
            except:
                pass  
        
        encoding = self.processor(
            image,
            question,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.processor.tokenizer.encode(
            answer,
            max_length=50,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        encoding["labels"] = labels.squeeze()
        
        encoding["question"] = question
        encoding["answer"] = answer
        encoding["image_path"] = image_path
        
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze()
        return encoding


class VQADataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image + text
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image_id = self.dataset[idx]['pid']
        image_path = f"/home/abdelrahman.elsayed/GP/vqa-finetune/Data/train_fill_in_blank/train_fill_in_blank/{image_id}/image.png"
        image = Image.open(image_path).convert("RGB")
        text = question
        
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length= 50, pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        # remove batch dimension
        for k,v in encoding.items():  encoding[k] = v.squeeze()
        return encoding