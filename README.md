# Fine-tuning Vision Transformer (ViT) for Multi-Task Face Recognition

This project demonstrates fine-tuning a pre-trained Vision Transformer (ViT) model for face recognition tasks, implementing multi-task learning to simultaneously classify both age groups and race categories.

## Project Overview

In this project, we fine-tune the `google/vit-base-patch16-224` Vision Transformer model on the UTKFace dataset to perform two classification tasks:

1. **Age Group Classification**:
   - Child/Youth (0-17)
   - Young Adult (18-39)
   - Middle-Aged (40-59)
   - Senior (60+)

2. **Race Classification**:
   - White
   - Black
   - Asian
   - Indian
   - Others

The project includes training, evaluation, and real-world testing components, demonstrating how a single model can be trained to recognize multiple attributes from facial images.

## Project Structure

```
vit-face-recognition/
├── data/                       # Directory for dataset storage
│   └── utkface/                # UTKFace dataset
├── fine_tuned_model/           # Directory to save fine-tuned model
├── results/                    # Directory for training results and logs
├── evaluation_results/         # Results from test set evaluation
├── real_world_results/         # Results from real-world image testing
├── src/
│   ├── train.py                # Main training script (multi-task learning)
│   ├── evaluate.py             # Script to evaluate on test data
│   ├── predict.py              # Script to test on real-world images
│   └── utils.py                # Utility functions
├── real_world_data/            # Directory for your test photos
│   └── test_images/
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

## Setup and Installation

1. Clone this repository:

```bash
git clone https://github.com/TheAnalystAbdalrahman/vit-face-recognition.git
cd vit-face-recognition
```

2. Create and activate a virtual environment (optional but recommended):

Option A: Using venv (standard Python virtual environment)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Option B: Using Conda (recommended for GPU setups)
```bash
conda create --name vit-env python=3.10 -y
conda activate vit-env
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/) from Hugging Face. The dataset contains over 23,000 face images with annotations for age, gender, and race. We chose this dataset for its:
- Diversity in demographics
- High-quality face images
- Rich attribute annotations
- Appropriate size for local training

## Usage

### 1. Training the Model

To train the model on the UTKFace dataset:

```bash
python src/train.py
```

This will:
- Load the UTKFace dataset from Hugging Face
- Preprocess the images with data augmentation
- Fine-tune the pre-trained ViT model with multi-task learning
- Save the fine-tuned model and evaluation results

### 2. Evaluating the Model

To evaluate the model on the test set:

```bash
python src/evaluate.py
```

This will:
- Load the fine-tuned model
- Evaluate it on the test split of UTKFace
- Generate detailed performance metrics and visualizations
- Create confusion matrices for both age and race classification

### 3. Testing on Real-World Images

To test the model on your own face images:

```bash
python src/predict.py --image_dir path/to/your/images
```

Place your test images in the `real_world_data/test_images/` directory or specify another directory with the `--image_dir` argument. The script will:
- Detect faces in your images
- Predict both age group and race for each face
- Generate visualizations with confidence scores
- Save results to CSV for analysis

## Results

### Training Performance

After fine-tuning, the model achieves:

**Age Classification:**
- Accuracy: 82.66%
- Precision: 82.60%
- Recall: 82.66%
- F1 Score: 82.55%

**Race Classification:**
- Accuracy: 84.24%
- Precision: 83.66%
- Recall: 84.24%
- F1 Score: 83.83%

**Combined Accuracy:** 83.45%

See the `results/` directory for detailed metrics and visualizations.

### Real-World Testing

The model was tested on personal photos to evaluate its performance on unseen data. As expected for a model not specifically trained on these individuals, performance varies based on image quality, lighting conditions, and demographic representation in the training data.

See the `real_world_results/` directory for predictions and visualizations.

## Model Architecture

This project implements a multi-task learning approach using Google's Vision Transformer (ViT) as the backbone:

- **Base Model**: `google/vit-base-patch16-224`
- **Image Size**: 224x224 pixels
- **Patch Size**: 16x16 pixels
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12

The custom model architecture includes:
1. A shared ViT backbone for feature extraction
2. Two separate classification heads:
   - Age classification head (4 classes)
   - Race classification head (5 classes)

This multi-task learning approach allows the model to leverage shared representations for both tasks, potentially improving performance compared to training separate models.

## Training Details

- **Batch Size**: 32 (optimized for GPU)
- **Learning Rate**: 2e-5 with cosine decay
- **Epochs**: 10
- **Optimizer**: AdamW with weight decay 0.01
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **Training Time**: 19.5 hours

## Acknowledgments

- The Vision Transformer model is from Google's ViT implementation in Hugging Face
- The UTKFace dataset is available through Hugging Face datasets
- This project was completed as part of the Deep Learning course term project

## Author

- [TheAnalystAbdalrahman](https://github.com/TheAnalystAbdalrahman)
