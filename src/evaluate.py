#!/usr/bin/env python3
# Evaluation script for fine-tuned multi-task ViT model for face recognition

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import random
from tqdm import tqdm
from safetensors.torch import load_file as safe_load



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create output directory
output_dir = "./evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# Define age group and race names
AGE_GROUP_NAMES = ['Child/Youth (0-17)', 'Young Adult (18-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
RACE_NAMES = ['White', 'Black', 'Asian', 'Indian', 'Others']

# Custom ViT model for multi-task learning
class ViTForMultiTaskClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_age_labels=4, num_race_labels=5):
        super(ViTForMultiTaskClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.vit.config.hidden_size
        
        # Age classification head
        self.age_classifier = nn.Linear(self.hidden_size, num_age_labels)
        
        # Race classification head
        self.race_classifier = nn.Linear(self.hidden_size, num_race_labels)
        
    def forward(self, pixel_values, labels=None, race_labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Get the [CLS] token representation
        
        # Age logits
        age_logits = self.age_classifier(cls_output)
        
        # Race logits
        race_logits = self.race_classifier(cls_output)
        
        # Calculate losses if labels are provided
        loss = None
        if labels is not None and race_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            age_loss = loss_fct(age_logits, labels)
            race_loss = loss_fct(race_logits, race_labels)
            loss = age_loss + race_loss  # Combined loss
        
        return {
            'loss': loss,
            'age_logits': age_logits,
            'race_logits': race_logits
        }

def map_age_to_group(age_idx):
    """Map age index to group"""
    # Get the age range name from the class index
    age_names = dataset['train'].features['age'].names
    age_class = age_names[age_idx]
    
    # Map age class to approximate age
    age_mapping = {
        '0-9': 5,
        '10-19': 15,
        '20-29': 25,
        '30-39': 35,
        '40-49': 45,
        '50-59': 55,
        '60-69': 65,
        '70+': 75
    }
    
    # Get the middle of the age range
    age = age_mapping[age_class]
    
    # Map to age group
    if age < 18:
        return 0  # Child/Youth
    elif age < 40:
        return 1  # Young Adult
    elif age < 60:
        return 2  # Middle-Aged
    else:
        return 3  # Senior

# Load model and processor
model_dir = "./fine_tuned_model"
processor = ViTImageProcessor.from_pretrained(model_dir)

# Load the base ViT model name
base_model_name = "google/vit-base-patch16-224"

# Create and load our custom model
model = ViTForMultiTaskClassification(base_model_name)

# Load the state dict
state_dict = safe_load(os.path.join(model_dir, "model.safetensors"))
model.load_state_dict(state_dict)


model.to(device)
model.eval()

# Load dataset
print("Loading UTKFace dataset...")
dataset = load_dataset("rixmape/utkface")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

# Process and evaluate test dataset
def evaluate_test_set(test_dataset, max_samples=5000):
    """Evaluate model on test dataset for both age and race"""
    # Limit test dataset size if needed
    if len(test_dataset) > max_samples:
        indices = random.sample(range(len(test_dataset)), max_samples)
        test_dataset = test_dataset.select(indices)
    
    true_age_labels = []
    predicted_age_labels = []
    true_race_labels = []
    predicted_race_labels = []
    image_ids = []
    age_confidences = []
    race_confidences = []
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # Process each test sample
    for i, example in enumerate(tqdm(test_dataset)):
        image = example['image'].convert('RGB')
        
        # Get true labels
        true_age_label = map_age_to_group(example['age'])
        true_race_label = example['race']  # UTKFace race labels are already indexed 0-4
        
        true_age_labels.append(true_age_label)
        true_race_labels.append(true_race_label)
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        processed_image = transform(image)
        
        # Get predictions
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get age predictions
        age_logits = outputs['age_logits']
        age_probabilities = torch.nn.functional.softmax(age_logits, dim=1)
        predicted_age = torch.argmax(age_probabilities, dim=1).cpu().numpy()[0]
        age_confidence = age_probabilities[0, predicted_age].cpu().numpy().item()
        
        # Get race predictions
        race_logits = outputs['race_logits']
        race_probabilities = torch.nn.functional.softmax(race_logits, dim=1)
        predicted_race = torch.argmax(race_probabilities, dim=1).cpu().numpy()[0]
        race_confidence = race_probabilities[0, predicted_race].cpu().numpy().item()
        
        predicted_age_labels.append(predicted_age)
        predicted_race_labels.append(predicted_race)
        image_ids.append(example['image_id'])
        age_confidences.append(age_confidence)
        race_confidences.append(race_confidence)
        
        # Save some example images for visualization
        if i < 10:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(np.array(example['image']))
            ax.set_title(f"Age - True: {AGE_GROUP_NAMES[true_age_label]}\nPred: {AGE_GROUP_NAMES[predicted_age]}, Conf: {age_confidence:.2f}\n\n"
                       f"Race - True: {RACE_NAMES[true_race_label]}\nPred: {RACE_NAMES[predicted_race]}, Conf: {race_confidence:.2f}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"example_{i}.png"))
            plt.close(fig)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'image_id': image_ids,
        'true_age_label': [AGE_GROUP_NAMES[label] for label in true_age_labels],
        'predicted_age_label': [AGE_GROUP_NAMES[label] for label in predicted_age_labels],
        'age_confidence': age_confidences,
        'true_race_label': [RACE_NAMES[label] for label in true_race_labels],
        'predicted_race_label': [RACE_NAMES[label] for label in predicted_race_labels],
        'race_confidence': race_confidences
    })
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Calculate metrics for age classification
    age_acc = accuracy_score(true_age_labels, predicted_age_labels)
    age_precision, age_recall, age_f1, _ = precision_recall_fscore_support(
        true_age_labels, predicted_age_labels, average='weighted', zero_division=0
    )
    
    # Calculate metrics for race classification
    race_acc = accuracy_score(true_race_labels, predicted_race_labels)
    race_precision, race_recall, race_f1, _ = precision_recall_fscore_support(
        true_race_labels, predicted_race_labels, average='weighted', zero_division=0
    )
    
    # Print classification reports
    print("\nAge Classification Report:")
    age_report = classification_report(true_age_labels, predicted_age_labels, target_names=AGE_GROUP_NAMES)
    print(age_report)
    
    print("\nRace Classification Report:")
    race_report = classification_report(true_race_labels, predicted_race_labels, target_names=RACE_NAMES)
    print(race_report)
    
    # Save reports to file
    with open(os.path.join(output_dir, 'classification_reports.txt'), 'w') as f:
        f.write("AGE CLASSIFICATION REPORT:\n")
        f.write(age_report)
        f.write("\n\nRACE CLASSIFICATION REPORT:\n")
        f.write(race_report)
    
    # Create age confusion matrix
    age_cm = confusion_matrix(true_age_labels, predicted_age_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(age_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=AGE_GROUP_NAMES,
                yticklabels=AGE_GROUP_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Age Group Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_confusion_matrix.png'))
    
    # Create race confusion matrix
    race_cm = confusion_matrix(true_race_labels, predicted_race_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(race_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=RACE_NAMES,
                yticklabels=RACE_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Race Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_confusion_matrix.png'))
    
    # Plot confidence distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(age_confidences, bins=20, kde=True)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Age Prediction Confidence Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(race_confidences, bins=20, kde=True)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Race Prediction Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distributions.png'))
    
    # Save metrics summary
    metrics = {
        'Age Accuracy': age_acc,
        'Age Precision': age_precision,
        'Age Recall': age_recall,
        'Age F1 Score': age_f1,
        'Race Accuracy': race_acc,
        'Race Precision': race_precision,
        'Race Recall': race_recall,
        'Race F1 Score': race_f1,
        'Combined Accuracy': (age_acc + race_acc) / 2
    }
    
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    
    print(f"\nEvaluation metrics:")
    print(f"Age Accuracy: {age_acc:.4f}")
    print(f"Age Precision: {age_precision:.4f}")
    print(f"Age Recall: {age_recall:.4f}")
    print(f"Age F1 Score: {age_f1:.4f}")
    print(f"\nRace Accuracy: {race_acc:.4f}")
    print(f"Race Precision: {race_precision:.4f}")
    print(f"Race Recall: {race_recall:.4f}")
    print(f"Race F1 Score: {race_f1:.4f}")
    print(f"\nCombined Accuracy: {(age_acc + race_acc) / 2:.4f}")
    
    print(f"\nResults saved to {output_dir}")
    
    return results_df, metrics

if __name__ == "__main__":
    # Run evaluation
    results, metrics = evaluate_test_set(test_dataset)