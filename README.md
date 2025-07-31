# HGCN: Hierarchical Graph Convolutional Network for Structured Long Document Classification

## Project Overview
This project implements a Hierarchical Graph Convolutional Network (HGCN) for structured long document classification and provides a complete workflow from raw data processing to model training. The project specifically focuses on the generation and utilization of enhanced embeddings to improve model performance.

## Complete Project Workflow

### 1. Environment Setup

### 2. Data Processing Workflow

#### 2.1 Named Entity Recognition (NER)
Extract entity information from text:

Output file: `ner.csv`

#### 2.2 Download Word Vector Model
Download model: `wikipedia2vec_models/wiki.en.vec`

#### 2.3 Extract Entity Vectors
Output file: `entity_embeddings.csv`

#### 2.4 Fusion Vectors into Original Data
Output files:
- `merged_kg_dataset.json`
- `enhanced_embeddings.json` (serves as base data for HGCN training)

#### 2.5 Generate Labels
Output file: `enhanced_embeddings_with_labels.json`

### 3. Model Training

#### 3.1 Configure Training Parameters
Modify the following parameters in `run_enhanced_embeddings_training.sh`:
- `CUDA_VISIBLE_DEVICES`: Specify GPU device
- `DATA_DIR`: Data file path
- `SAVE_PATH`: Model saving path
- Other training parameters (batch size, learning rate, etc.)

#### 3.2 Run Training Script

## Project Structure
```plaintext
HGCN-master/
├── src/
│   ├── train_enhanced_embeddings.py  # Enhanced embedding training script
│   ├── train.py                      # Regular training script
│   └── args.py                       # Parameter configuration
├── common/                           # Common modules
│   ├── evaluators/                   # Evaluators
│   └── trainers/                     # Trainers
├── datasets/                         # Dataset processing
│   ├── bert_processors/              # BERT processors
│   └── bow_processors/               # BoW processors
├── utils/                            # Utility functions
├── ner.py                            # NER processing script
├── download_wikipedia2vec.py         # Script to download word vector model
├── extract_vector.py                 # Script to extract entity vectors
├── Fusion_vector.py                  # Vector fusion script
├── generate_labels.py                # Script to generate labels
├── run_enhanced_embeddings_training.sh  # Training execution script
└── model_checkpoints/                # Model saving directory

# Data Format Description

## Input Data Format
`enhanced_embeddings.json` contains the following fields:
- `id`: Unique identifier
- `title`: Document title
- `kg_embeddings`: Knowledge graph embedding vectors
- `kg_entities`: List of related entities
- `enhanced_vector`: Enhanced vector
- `vector_dim`: Vector dimension

## Output Data Format
Trained models are saved in `./model_checkpoints/enhanced_embeddings/`  
Includes model weights, configuration files, and training logs

# Notes
- Ensure all data processing steps are completed before running the training script
- Adjust batch size and sequence length according to your hardware configuration
- CUDA support is required during training (GPU usage recommended)
- To modify the model architecture, refer to `src/train_enhanced_embeddings.py`

# Troubleshooting
- CUDA out of memory: Try reducing batch size or sequence length
- Poor model convergence: Adjust learning rate or increase training epochs
- Data format errors: Check if input JSON files meet requirements
