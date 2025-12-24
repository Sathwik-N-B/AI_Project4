# AI Project 4: Mushroom Classification

## Overview
This project implements a machine learning pipeline for classifying mushrooms as **edible** or **poisonous** using multiple transformer-based models with Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Features
- **Data Preprocessing**: Converts raw mushroom feature codes into human-readable descriptions
- **Multiple Models**: Implements classification using DistilBERT, BERT, and GPT-2 architectures
- **LoRA Fine-Tuning**: Uses Low-Rank Adaptation for efficient model training
- **Multiple Prompt Styles**: 
  - Short prompts
  - Verbose prompts
  - Chain-of-Thought (CoT) prompts
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and visualization

## Requirements
```bash
pip install transformers datasets peft accelerate bitsandbytes pandas scikit-learn matplotlib torch
```

## Dataset
The project uses the UCI Mushroom dataset (`mushrooms_1.csv`) with features including:
- Cap shape, surface, and color
- Odor characteristics
- Gill attributes
- Stalk properties
- Ring and veil characteristics
- Habitat and population information

## Models Implemented

### 1. DistilBERT (Classification)
- Base model: `distilbert-base-uncased`
- LoRA configuration with rank 8
- Target modules: `q_lin`, `v_lin`

### 2. BERT (Classification)
- Base model: `bert-base-uncased`
- LoRA rank 8
- Target modules: `query`, `value`

### 3. GPT-2 (Generative)
- Base model: `gpt2`
- LoRA rank 16
- Target modules: `c_attn`
- 4-bit quantization for memory efficiency

## Usage

### Running the Jupyter Notebook
```bash
jupyter notebook AI_Project4_snellikoppab.ipynb
```

### Running the Python Script
```bash
python ai_project4_snellikoppab.py
```

## Project Structure
```
AI_Project4/
│
├── AI_Project4_snellikoppab.ipynb    # Main Jupyter notebook
├── ai_project4_snellikoppab.py       # Python script version
├── mushrooms_1.csv                    # Input dataset (required)
└── README.md                          # This file
```

## Workflow

1. **Data Preparation**
   - Load mushroom dataset
   - Map abbreviated features to descriptive text
   - Generate training/testing datasets with different prompt styles

2. **Model Training**
   - Apply LoRA adapters to transformer models
   - Fine-tune on mushroom classification task
   - Train with optimized hyperparameters

3. **Evaluation**
   - Calculate accuracy metrics
   - Generate confusion matrices
   - Compare performance across models and prompt styles

4. **Analysis**
   - Visualize results
   - Compare model performances
   - Identify best performing configurations

## Expected Outputs
- Training and test JSONL files for each prompt style
- Trained model checkpoints with LoRA adapters
- Evaluation metrics and confusion matrices
- Performance comparison visualizations

## Configuration
```python
DATA_PATH = "/content/mushrooms_1.csv"  # Update this path
RANDOM_SEED = 42
TEST_SIZE = 0.2
```

## Key Components

### Feature Mapping
Converts single-character codes to descriptive text:
- `'e'` → `'edible'`
- `'p'` → `'poisonous'`
- Cap shape codes → descriptive terms (bell, conical, convex, etc.)

### LoRA Configuration
Efficient fine-tuning with reduced parameters:
- Lower rank (r=8 or r=16)
- Targeted module adaptation
- Dropout for regularization

### Training Arguments
- Learning rate: 3e-4
- Batch size: 16
- Weight decay: 0.01
- Automatic mixed precision (FP16)

## Performance Metrics
The project evaluates models using:
- Accuracy score
- Confusion matrix
- Training/validation loss curves
- Cross-model performance comparison

## Author
Sathwik Nellikoppa Basavaraja

## Notes
- Ensure the mushroom dataset is available at the specified `DATA_PATH`
- GPU recommended for faster training (CUDA support)
- Models use 4-bit quantization for memory efficiency
- Results are saved in the working directory

## License
This is an academic project for educational purposes.
