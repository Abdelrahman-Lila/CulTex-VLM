import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from nlpaug.augmenter.word import ContextualWordEmbsAug

class VQADataset(Dataset):
    def __init__(self, data, processor, is_train=False, data_type="csv"):
        """
        Unified VQA Dataset class
        Args:
            data: Can be pandas DataFrame or HF dataset
            processor: BLIP processor
            is_train: Whether to use data augmentation
            data_type: "csv" or "hf" (huggingface dataset)
        """
        self.data = data
        self.processor = processor
        self.is_train = is_train
        self.data_type = data_type
        
        if self.is_train:
            self.aug = ContextualWordEmbsAug(
                model_path='bert-base-uncased', 
                action="substitute",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

    def __len__(self):
        return len(self.data)

    def _get_item_data(self, idx):
        """Helper method to extract fields based on data type"""
        if self.data_type == "csv":
            row = self.data.iloc[idx]
            return {
                "question": str(row["Question"]),
                "answer": str(row["Answer"]),
                "image_path": row["img_path"]
            }
        elif self.data_type == "hf":
            item = self.data[idx]
            return {
                "question": item["question"],
                "answer": item["answer"],
                "image_path": f"/home/abdelrahman.elsayed/GP/vqa-finetune/Data/train_fill_in_blank/train_fill_in_blank/{item['pid']}/image.png"
            }
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def __getitem__(self, idx):
        try:
            # Get raw data based on data type
            item_data = self._get_item_data(idx)
            question = item_data["question"]
            answer = item_data["answer"]
            image_path = item_data["image_path"]

            # Load image with error handling
            image = Image.open(image_path).convert("RGB")
            
            # Data augmentation for training
            if self.is_train and random.random() < 0.5:
                try:
                    question = self.aug.augment(question)[0]  # Augment returns list
                except:
                    pass

            # Process inputs
            inputs = self.processor(
                images=image,
                text=question,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Process answers
            answer_inputs = self.processor(
                text=answer,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "pixel_values": inputs.pixel_values.squeeze(),
                "input_ids": inputs.input_ids.squeeze(),
                "attention_mask": inputs.attention_mask.squeeze(),
                "labels": answer_inputs.input_ids.squeeze(),
                "question": question,
                "answer": answer,
                "image_path": image_path
            }
            
        except Exception as e:
            print(f"Error loading index {idx}: {str(e)}")
            return self.__getitem__(random.randint(0, len(self)-1))  # Return random sample on error

def collate_fn(batch):
    """Handles both tensor and non-tensor data"""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "image_path": [item["image_path"] for item in batch]
    }