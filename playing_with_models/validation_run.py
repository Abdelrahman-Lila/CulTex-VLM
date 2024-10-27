# Install dependencies
os.system("pip install -r requirements.txt")


import torch
from PIL import Image
import evaluate
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import (
    Blip2ForConditionalGeneration,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    PaliGemmaForConditionalGeneration,
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
)
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from qwen_vl_utils import process_vision_info
import pandas as pd
from datasets import Dataset
import os
import gc
import yaml
from groq import Groq
import tensorflow_hub as hub
import tensorflow as tf
import re
import time


class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, model,processor, device, prompt=None):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.model = model
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        caption = item["caption_english"]
        if config["model"]["arch"] == "deepseek":
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Describe each stage of this image.",
                    "images": [image_path],
                },
                {"role": "Assistant", "content": ""},
            ]
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(self.model.device)
            inputs = self.model.prepare_inputs_embeds(**prepare_inputs)
        elif config["model"]["arch"] == "qwen":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image",
                        "image": image_path},
                        {"type": "text",
                        "text": "Describe this image."},
                    ],
                },
            ]
            # load images and prepare for inputs
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
        else:
            image = Image.open(image_path).convert("RGB")
            if self.prompt is not None:
                print(f"\nUsing the prompt {self.prompt} with ImageCaptioningDataset Class")
                inputs = self.processor(
                    images=[image], text=self.prompt, return_tensors="pt"
                ).to(self.device, torch.bfloat16)
            else:
                inputs = self.processor(images=[image], return_tensors="pt").to(
                    self.device, torch.bfloat16
                )

            # ensure Everything on the right device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return {"image_path": image_path, "inputs": inputs, "caption_english": caption}


with open(
    "/home/abdelrahman.elsayed/GP/playing_with_models/model_config.yaml", "r"
) as f:
    config = yaml.safe_load(f)

print("\nThe current config", config)


class ModelValidator:
    def __init__(self, config, model_path=None, processor_path=None, model=None, processor=None ,prompt=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.last_request_time = time.time()
        self.min_interval = 60 / 30  # 30 requests per minute
        # Initialize metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = None

        if self.config["params"]["semantic"] == "senEnc":
            self.embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            # Load model and processor
            if self.config["model"]["arch"] == "BLIP":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_path,
                )
                self.processor = BlipProcessor.from_pretrained(processor_path)
                self.model = self.model.to(self.device)
            elif self.config["model"]["arch"] == "cham":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                self.model = ChameleonForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map="cuda"
                ).to(self.device)
                self.processor = ChameleonProcessor.from_pretrained(processor_path)
            elif self.config["model"]["arch"] == "idefics":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(processor_path)
            elif self.config["model"]["arch"] == "deepseek":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                self.processor = VLChatProcessor.from_pretrained(model_path)
                self.tokenizer = self.processor.tokenizer
                vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self.model = vl_gpt.to(torch.bfloat16).cuda().eval()
                # self.model.gradient_checkpointing_enable()
            elif self.config["model"]["arch"] == "Pali":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                model_id = "google/paligemma-3b-mix-224"
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(
                    self.device
                )
                self.processor = AutoProcessor.from_pretrained(model_id)
            elif self.config["model"]["arch"] == "quen":
                assert model_path is not None, "Provide the model path"
                assert processor_path is not None, "Provide the processor path"
                model_id = "Qwen/Qwen2-VL-2B-Instruct"
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id).to(
                    self.device
                )
                self.processor = AutoProcessor.from_pretrained(model_id)
            else:
                # Load default model
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

        self.model.eval()  # Set model to evaluation mode

    def generate_caption(self, image_path, max_new_tokens=300):
        """Generate caption for a single image"""
        if config["model"]["arch"] == "deepseek":
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Describe each stage of this image.",
                    "images": [image_path],
                },
                {"role": "Assistant", "content": ""},
            ]
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(self.model.device)
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            # run the model to get the response
            try:
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )

                answer = self.tokenizer.decode(
                    outputs[0].cpu().tolist(), skip_special_tokens=True
                )
                return answer
            except Exception as e:
                print(f"Error generating caption for {image_path}: {str(e)}")
                return ""
        elif self.config["model"]["arch"] == "qwen":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image",
                        "image": image_path},
                        {"type": "text",
                        "text": "Describe this image."},
                    ],
                },
              ]
            try:
            # load images and prepare for inputs
                text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(conversation)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        max_length= max_new_tokens,
                    )
                outputs_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                caption = self.processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return caption
            except Exception as e:
                print(f"Error generating caption for {image_path}: {str(e)}")
                return ""
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                if self.prompt is not None:
                    print(f"\nUsing the prompt {self.prompt} with ModelValidator Class")
                    inputs = self.processor(
                        images=[image], text=self.prompt, return_tensors="pt"
                    ).to(self.device, torch.bfloat16)
                else:
                    inputs = self.processor(image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        max_length=max_new_tokens,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0,
                    )

                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                return caption
            except Exception as e:
                print(f"Error generating caption for {image_path}: {str(e)}")
                return ""

    def calculate_semantic_similarity(self, generated_caption, reference_caption):
        """Calculate semantic similarity using Groq's API."""
        # Get the current time to check our rate
        now = time.time()
        time_since_last_request = now - self.last_request_time
        # If we're sending requests too quickly, sleep for the remaining time
        if time_since_last_request < self.min_interval:
            time.sleep(self.min_interval - time_since_last_request)

        try:
            client = Groq(
                api_key="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"
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Compare the semantic similarity between the following two texts and return a score:\n\nText 1: {generated_caption}\n\nText 2: {reference_caption}\n\nThe score should be between 0 (no similarity) and 1 (identical meaning) and you should answer with the number only.",
                    }
                ],
                model="llama3-8b-8192",
            )

            response = chat_completion.choices[0].message.content

            print(response)

            pattern = r"0\.\d+|1\.0+"

            matches = re.findall(pattern, response)
            if len(matches) > 0:
                return float(matches[0])
            else:
                print("EMPTY Response\n")
                return 0.0
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0.0
        finally:
            self.last_request_time = time.time()



    def sentence_encoding_semantic_similarity(self, generated_caption, reference_caption):
        embeddings = self.embed([generated_caption, reference_caption])

        cosine_similarity = -1 * tf.keras.losses.cosine_similarity(embeddings[0], embeddings[1]).numpy()
        return cosine_similarity
        


    def calculate_metrics(
        self, generated_caption, reference_caption, semantic="None"
    ):
        """Calculate multiple metrics for a single caption pair"""
        # Prepare inputs
        prediction = generated_caption.lower().split()
        reference = [reference_caption.lower().split()]

        if semantic == "None":
            smoothing = SmoothingFunction().method1

            # Calculate BLEU scores
            bleu_1 = sentence_bleu(
                reference,
                prediction,
                weights=(1, 0, 0, 0),
                smoothing_function=smoothing,
            )
            bleu_2 = sentence_bleu(
                reference,
                prediction,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothing,
            )
            bleu_3 = sentence_bleu(
                reference,
                prediction,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smoothing,
            )
            bleu_4 = sentence_bleu(
                reference,
                prediction,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )

            # Calculate ROUGE scores
            rouge_scores = self.scorer.score(
                reference_caption.lower(), generated_caption.lower()
            )

            # Calculate METEOR score
            meteor_score = self.meteor.compute(
                predictions=[generated_caption.lower()],
                references=[[reference_caption.lower()]],
            )["meteor"]
            return {
                "bleu_1": bleu_1,
                "bleu_2": bleu_2,
                "bleu_3": bleu_3,
                "bleu_4": bleu_4,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
                "meteor": meteor_score,
            }
        elif semantic == "Grog":
            # get our new guy
            semantic_similarity = self.calculate_semantic_similarity(
                generated_caption, reference_caption
            )
            return {"semantic_similarity": semantic_similarity}
        elif semantic == "senEnc":
            # get our new guy
            semantic_similarity = self.sentence_encoding_semantic_similarity(
                generated_caption, reference_caption
            )
            return {"semantic_similarity": semantic_similarity}

    def save_qualitative_results(self, all_metrics, model_name, num_samples=10):
        """Save generated captions and reference texts in a qualitative folder"""
        qualitative_dir = "qualitative"
        os.makedirs(qualitative_dir, exist_ok=True)

        qualitative_file = os.path.join(
            qualitative_dir, f"{model_name}_qualitative_results.txt"
        )

        with open(qualitative_file, "w") as f:
            f.write(f"Qualitative Results for Model: {model_name}\n")
            f.write("=" * 40 + "\n\n")

            for idx, item in enumerate(all_metrics[:num_samples]):
                f.write(f"Image Path: {item['image_path']}\n")
                f.write(f"Reference Caption: {item['reference_caption']}\n")
                f.write(f"Generated Caption: {item['generated_caption']}\n")
                f.write("-" * 40 + "\n\n")

    def validate_dataset(self, eval_dataset, model_name):
        """Validate model on entire evaluation dataset"""
        all_metrics = []

        for item in tqdm(eval_dataset, desc="Validating"):
            torch.cuda.empty_cache()
            generated_caption = self.generate_caption(item["image_path"])
            metrics = self.calculate_metrics(
                generated_caption,
                item["caption_english"],
                semantic=config["params"]["semantic"],
            )

            # Store results
            result = {
                "image_path": item["image_path"],
                "reference_caption": item["caption_english"],
                "generated_caption": generated_caption,
                **metrics,
            }
            all_metrics.append(result)

        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([item[metric] for item in all_metrics])
            for metric in all_metrics[0].keys()
            if metric not in ["image_path", "reference_caption", "generated_caption"]
        }

        # Save qualitative results
        self.save_qualitative_results(all_metrics, model_name)

        return all_metrics, avg_metrics
        return all_metrics, avg_metrics

    def save_results(self, all_metrics, avg_metrics, output_path="validation_results"):
        """Save validation results to files"""
        os.makedirs(output_path, exist_ok=True)

        # Save detailed results
        df_detailed = pd.DataFrame(all_metrics)
        df_detailed.to_csv(
            os.path.join(output_path, "detailed_results.csv"), index=False
        )

        # Save average metrics
        df_avg = pd.DataFrame([avg_metrics])
        df_avg.to_csv(os.path.join(output_path, "average_metrics.csv"), index=False)

        # Save results in readable format
        with open(os.path.join(output_path, "validation_report.txt"), "w") as f:
            f.write("=== Model Validation Report ===\n\n")
            f.write("Average Metrics:\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")


def create_image_caption_pairs(df, image_root_folder, batch_size=10):
    data = []

    for idx, row in df.iterrows():
        subfolder_name = row["exhibits"]
        caption_english = row["Text in English"]

        subfolder_path = os.path.join(image_root_folder, f"{idx+1}_{subfolder_name}")

        if os.path.exists(subfolder_path):
            image_files = [
                f
                for f in os.listdir(subfolder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            for image_file in image_files:
                data.append(
                    {
                        "image_path": os.path.join(subfolder_path, image_file),
                        "caption_english": caption_english,
                    }
                )

        # Process in batches to save memory
        if len(data) >= batch_size:
            yield data
            data = []

    if data:  # Yield remaining data
        yield data


def main():
    excel_file = "/home/abdelrahman.elsayed/GP/dataset/data set.xlsx"
    image_root_folder = "/home/abdelrahman.elsayed/GP/dataset/training_set"

    df = pd.read_excel(excel_file)

    all_data = []
    for data_chunk in create_image_caption_pairs(df, image_root_folder):
        all_data.extend(data_chunk)

        # Clear memory periodically
        if len(all_data) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Create custom datasets
    dataset = Dataset.from_list(all_data)
    train_test_split = dataset.train_test_split(test_size=0.2)

    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    processor_path = "deepseek-ai/deepseek-vl-7b-chat"
    # text = None
    # text = "Describe this ancient item in detail to the best of your knowledge. Include information about its origin, historical significance, materials used, craftsmanship, and any known uses or cultural importance. Provide as much context as possible to help understand its place in history."
    text = "<image>Describe this ancient item in detail to the best of your knowledge."
    validator = ModelValidator(
        model_path=model_path, processor_path=processor_path, config=config, prompt=text
    )

    eval_dataset = ImageCaptioningDataset(
        train_test_split["test"],validator.model, validator.processor, validator.device, prompt=text
    )

    print("\nThe eval dataset", len(eval_dataset))

    if validator.prompt is not None:
        model_name = validator.model.config._name_or_path.split("/")[-1] + "_prompted"
    else:
        model_name = validator.model.config._name_or_path.split("/")[-1]

    if config["params"]["semantic"]:
        model_name += "_semantic"

    # Run validation
    all_metrics, avg_metrics = validator.validate_dataset(
        eval_dataset, model_name=model_name
    )

    # Save results
    output_path = f"{model_name}_results"
    print("\nSaving results in: ", output_path)
    validator.save_results(all_metrics, avg_metrics, output_path=output_path)

    print("\nValidation Results:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
