# CulTex-VLM

*Bridging the gap between AI and cultural understanding*

A comprehensive framework for evaluating and fine-tuning Vision-Language Models (VLMs) on cultural heritage datasets, with a special focus on Egyptian cultural artifacts and museum collections.
## CulTeX-VLM (RAG) DEMO  
[Watch the demo video](https://github.com/python-arch/CulTex-VLM/blob/master/videos/CulTeXVLMDemo.mp4?raw=true)

## MCP DEMO  
[Watch the demo video](https://github.com/python-arch/CulTex-VLM/blob/master/videos/MCP%20Demo.mp4?raw=true)

## 🎯 Overview

- **Identifying cultural blind spots** in existing VLMs through systematic evaluation
- **Creating culturally-aware datasets** from scratch using innovative data collection methods  
- **Developing specialized fine-tuning techniques** that respect cultural context
- **Providing a robust evaluation framework** that goes beyond standard metrics

We're not just building another AI model – we're creating a bridge between artificial intelligence and human cultural understanding.

## ✨ Key Features
Name

- **🔍 Multi-Model Benchmarking**: Comprehensive head-to-head evaluation of 6 different Vision-Language Models
- **🏛️ Cultural Heritage Expertise**: Specially crafted for Egyptian cultural heritage using the National Museum of Egyptian Civilization dataset
- **🎯 Three Fine-tuning Flavors**: Choose between Full fine-tuning (maximum power), LoRA (efficient adaptation), or RAG (external knowledge integration)
- **📊 Comprehensive Evaluation**: Multiple metrics that actually matter - BLEU, ROUGE, METEOR, BERTScore, and Exact Match
- **⚖️ Fair Play Protocol**: Matched-size evaluation ensures no model gets an unfair advantage
- **🤖 Automated Data Pipeline**: From concept to dataset in one go - scrape, validate, and structure cultural data automatically

## 🔧 Installation

```bash
git clone https://github.com/python-arch/CulTex-VLM.git
cd CulTex-VLM
pip install -r requirements.txt
```

## 📊 Evaluation Metrics
*Because one metric is never enough*

Our evaluation goes beyond simple accuracy to capture the nuances of cultural understanding:

### 📝 Text Generation Quality
- **BLEU-4**: Measures how well generated text matches reference answers (n-gram precision)
- **ROUGE-1/2/L**: Focuses on recall - does the model capture the key information? (unigrams, bigrams, longest common subsequences)
- **METEOR**: The sophisticated metric that considers stemming, synonyms, and word order - perfect for cultural nuances

### 🎯 Model Performance  
- **Cross-Entropy Loss**: The training objective - lower is better
- **BERTScore-F₁**: Uses RoBERTa-large to measure semantic similarity at the sentence level
- **Exact Match (EM)**: Sometimes you need the answer to be exactly right - strict string matching after normalization

*Why multiple metrics?* Cultural content is complex. A model might get the gist right (good ROUGE) but miss specific cultural terms (bad Exact Match). Our comprehensive evaluation captures these subtleties.

## 🏗️ Fine-tuning Strategies
*Choose your adventure based on your resources and needs*

### 1. 💪 Full Fine-tuning
*"Go big or go home"*
- Updates all 384.9M parameters of BLIP
- Maximum adaptation capability for your specific cultural domain
- Perfect when you have the computational resources and want the best results
- Higher computational requirements but potentially superior performance

### 2. ⚡ Low-Rank Adaptation (LoRA)
*"Smart efficiency meets effectiveness"*
- The sweet spot between performance and efficiency
- Injects tiny low-rank matrices (r=8) into attention projections
- Only trains 0.077% of parameters while keeping the massive backbone frozen
- Applied strategically to query, key, and value projections in both image encoder and language decoder
- Perfect for researchers with limited GPU resources

### 3. 🧠 Retrieval-Augmented Generation (RAG)
*"Knowledge without training"*
- Zero fine-tuning approach - your base model stays untouched
- Integrates external cultural knowledge through intelligent retrieval
- Uses CLIP-based similarity search with ChromaDB FAISS index
- Retrieves top-3 most relevant captions as context for each query
- Ideal when you want to add cultural knowledge without model modification

## 📚 Dataset Construction

We created our cultural heritage datasets using two innovative approaches, starting from the ground up since existing VQA datasets lack cultural domain coverage.

### 🔍 Data Collection Pipeline

Our journey began with building a robust automated data collection system specifically designed for cultural content:

**What we collected:**
- 270+ Egyptian cultural entities (from "Tutankhamun" to "Koshari" to "Siwa Oasis")
- Over 5,000 high-quality images scraped from DuckDuckGo
- Rich textual descriptions from Wikipedia and Encyclopedia Britannica
- Structured, organized data ready for model training

**How we did it:**
1. **Smart Image Scraping**: Used Playwright to automate DuckDuckGo image search (avoiding Google's restrictions)
2. **Rich Context Gathering**: Pulled comprehensive background information from Wikipedia API
3. **Enhanced Coverage**: Supplemented with Encyclopedia Britannica's chatbot for deeper cultural insights
4. **Quality Organization**: Each cultural concept gets its own folder with 20 images and detailed background information

### 🎯 VQA Dataset Creation

From our collected cultural data, we developed two distinct approaches to create question-answer pairs:

#### Approach 1: VQ2A (Caption-to-Questions) 
*"Let the images tell their stories"*

Starting with 50 carefully curated image-caption pairs, we used an automated pipeline inspired by Visual Question Generation with Question Answering validation:

1. **Smart Answer Extraction**: Used spaCy to intelligently parse captions and extract potential answers (nouns, verbs, adjectives, spatial relationships)
2. **Natural Question Generation**: Leveraged DeepSeek API to create human-like questions from extracted answers
3. **Quality Validation**: Implemented F2 score filtering to ensure question-answer consistency
4. **Impressive Scale**: Transformed 50 image-caption pairs into 1,706 validated VQA triplets (34× expansion!)

#### Approach 2: CultureVLM (Template-Based)
*"Systematic cultural exploration"*

A more structured approach focusing on cultural categories:

1. **Cultural Classification**: Organized images into meaningful categories (food, clothing, architecture, rituals)
2. **Expert-Crafted Templates**: Created category-specific question templates like "What traditional dance is this?" or "What type of architecture is shown?"
3. **AI-Enhanced Diversity**: Used Gemini API to generate additional natural variations
4. **Human Review**: Manual validation to ensure cultural accuracy and authenticity

### 📊 Final Dataset Overview

| Dataset | Size | Train | Validation | Description |
|---------|------|-------|------------|-------------|
| **VQ2A Original** | 2,750 | 2,475 | 275 | Caption-generated questions |
| **VQ2A Cleaned** | 1,705 | 1,534 | 171 | Quality-filtered version |
| **CultureVLM** | 1,800 | 1,600 | 200 | Template-based cultural questions |
| **IconDomainVQA** | 4,364 | 3,928 | 436 | Domain-agnostic baseline |

### 🎲 Fair Evaluation Protocol
We implemented a matched-size protocol to ensure fair model comparisons:
- **Global cap**: 2,750 samples (matching our largest custom dataset)
- **Smart sampling**: Uniform sampling for larger datasets to maintain representativeness
- **Consistent splits**: 90:10 train/validation with fixed random seed (42)
- **Preserved diversity**: Maintains each dataset's intrinsic cultural distribution

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/python-arch/CulTex-VLM.git
cd CulTex-VLM
pip install -r requirements.txt
```

### Evaluate a Model in 3 Lines
```python
from cultex_vlm import VLMEvaluator

evaluator = VLMEvaluator(model_name="blip", dataset="cultured_vlm")
results = evaluator.evaluate(metrics=["bleu", "rouge", "bertscore"])
print(f"Your model scored: {results['bertscore_f1']:.3f} on cultural understanding!")
```

### Fine-tune with LoRA (Efficient Training)
```python
from cultex_vlm import BLIPFineTuner

# The sweet spot between performance and efficiency
tuner = BLIPFineTuner(
    method="lora",           # Efficient training
    rank=8,                 # Low-rank adaptation
    learning_rate=2e-5,     # Gentle learning
    cultural_focus=True     # Our secret sauce
)

# Transform your model in minutes, not hours
model = tuner.train(dataset_path="data/egyptian_heritage", epochs=10)
```

### Try RAG (Zero Training Required!)
```python
from cultex_vlm import RAGPipeline

# No training needed - just add knowledge!
rag = RAGPipeline(
    clip_model="ViT-B/32",
    k_neighbors=3,
    knowledge_base="egyptian_cultural_database"
)

# Ask anything about cultural artifacts
answer = rag.generate(
    image_path="mysterious_artifact.jpg", 
    question="What historical period is this artifact from?"
)
print(f"Cultural insight: {answer}")
```

## 📊 Results Structure

```
results/
├── evaluation/
│   ├── bleu_scores.json
│   ├── rouge_scores.json
│   └── meteor_scores.json
├── models/
│   ├── full_finetune/
│   ├── lora/
│   └── rag/
└── logs/
    └── training_logs.txt
``

