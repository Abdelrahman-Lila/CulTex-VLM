import os
from datasets import Dataset
from PIL import Image

class ArrowDataProcessor:
    def __init__(self, dataset_folder, temp_image_folder="/tmp/arrow_images"):
        """
        Initializes the processor with the dataset folder.
        :param dataset_folder: Folder containing Arrow files.
        :param temp_image_folder: Temporary folder for saving processed images.
        """
        self.dataset_folder = dataset_folder
        self.temp_image_folder = temp_image_folder
        os.makedirs(self.temp_image_folder, exist_ok=True)

    def process(self):
        """
        Processes the Arrow files and converts them into a HuggingFace Dataset.
        :return: A HuggingFace Dataset with image paths and captions.
        """
        data = []  # Prepare data for the Dataset

        for root, _, files in os.walk(self.dataset_folder):
            for file in files:
                if file.endswith('.arrow'):
                    arrow_file_path = os.path.join(root, file)
                    print(f"Processing file: {arrow_file_path}")
                    dataset = Dataset.from_file(arrow_file_path)

                    for entry in dataset:
                        image = entry['image']  # Assumes 'image' is a PIL.Image object
                        captions = entry.get('annotations_captions', [""])

                        # Save image temporarily
                        image_id = entry.get('id', len(data))  # Fallback to index
                        image_path = os.path.join(self.temp_image_folder, f"{image_id}.jpg")
                        image.save(image_path)

                        # Add entry to data
                        data.append({
                            "image_path": image_path,
                            "caption_english": captions[0]  # Use the first caption
                        })

        return Dataset.from_list(data)

    def cleanup(self):
        """
        Cleans up temporary images stored during processing.
        """
        for root, _, files in os.walk(self.temp_image_folder):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(self.temp_image_folder)
