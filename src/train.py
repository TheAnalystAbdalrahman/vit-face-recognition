#!/usr/bin/env python3
# Fine-tuning Vision Transformer (ViT) for Face Recognition with Multi-Task Learning
# Deep Learning Term Project - Age and Race Classification

import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    ViTModel,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import random

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create necessary directories
os.makedirs("./results", exist_ok=True)
os.makedirs("./fine_tuned_model", exist_ok=True)

# Define MLflow experiment
mlflow.set_experiment("vit-face-recognition")

# Configuration - Increased for full training
model_name = "google/vit-base-patch16-224"
batch_size = 32 if torch.cuda.is_available() else 8  # Increased batch size for GPU
num_epochs = 10 if torch.cuda.is_available() else 2  # Increased epochs for better learning
learning_rate = 2e-5  
max_samples = 15000 if torch.cuda.is_available() else 1000  # Use most of the dataset
output_dir = "./results"
model_save_dir = "./fine_tuned_model"
test_size = 0.2  # Standard test split

# Define class names
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

# Initialize MLflow run
with mlflow.start_run(run_name="vit-face-recognition-utkface-full"):
    
    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_samples", max_samples)
    mlflow.log_param("test_size", test_size)

    # Load the UTKFace dataset
    print("Loading UTKFace dataset...")
    dataset = load_dataset("rixmape/utkface")
    
    # Display dataset information
    print(f"Dataset info: {dataset}")
    print(f"Features: {dataset['train'].features}")
    
    # Create train/test split since dataset doesn't have one
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)
    print(f"Train/Test split created: {dataset}")
    
    # Define image processor/feature extractor
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Map age to age group
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
    
    # We'll use race categories directly
    def get_race_label(race_idx):
        """Get race label"""
        return race_idx  # UTKFace race labels are already indexed from 0-4
    
    # Enhanced image processing pipeline for multi-task learning
    def process_example(example):
        # Convert to RGB if necessary (some UTKFace images might be grayscale)
        image = example['image'].convert('RGB')
        
        # Apply more preprocessing for better results
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),  # Data augmentation
            transforms.RandomRotation(degrees=10),   # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image)
        
        # Map age to group
        age_group = map_age_to_group(example['age'])
        
        # Get race label
        race_label = get_race_label(example['race'])
        
        return {
            'pixel_values': image_tensor.numpy(),  # Store as numpy array to avoid issues
            'label': age_group,                    # Age group label
            'race_label': race_label               # Race label
        }
    
    # Apply processing to the dataset for multi-task learning
    def process_dataset(dataset_split):
        # Limit dataset size
        if len(dataset_split) > max_samples:
            indices = random.sample(range(len(dataset_split)), max_samples)
            dataset_split = dataset_split.select(indices)
        
        processed_dataset = dataset_split.map(
            process_example,
            remove_columns=['image_id', 'image', 'age', 'gender', 'race']
        )
        return processed_dataset
    
    processed_train = process_dataset(dataset['train'])
    processed_test = process_dataset(dataset['test'])
    
    # Print dataset statistics
    print(f"Processed train dataset size: {len(processed_train)}")
    print(f"Processed test dataset size: {len(processed_test)}")
    
    # Visualize label distributions for both age and race
    train_age_labels = [example['label'] for example in processed_train]
    test_age_labels = [example['label'] for example in processed_test]
    train_race_labels = [example['race_label'] for example in processed_train]
    test_race_labels = [example['race_label'] for example in processed_test]
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=train_age_labels)
    plt.title('Training Set Age Distribution')
    plt.xlabel('Age Group')
    plt.xticks(range(4), AGE_GROUP_NAMES, rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=test_age_labels)
    plt.title('Test Set Age Distribution')
    plt.xlabel('Age Group')
    plt.xticks(range(4), AGE_GROUP_NAMES, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    mlflow.log_artifact(os.path.join(output_dir, 'age_distribution.png'))
    
    # Race distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=train_race_labels)
    plt.title('Training Set Race Distribution')
    plt.xlabel('Race')
    plt.xticks(range(5), RACE_NAMES, rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=test_race_labels)
    plt.title('Test Set Race Distribution')
    plt.xlabel('Race')
    plt.xticks(range(5), RACE_NAMES, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_distribution.png'))
    mlflow.log_artifact(os.path.join(output_dir, 'race_distribution.png'))
    
    # Define multi-task model
    print("Loading pre-trained ViT model for multi-task learning...")
    model = ViTForMultiTaskClassification(
        model_name,
        num_age_labels=4,  # Age groups: Child, Young Adult, Middle-Aged, Senior
        num_race_labels=5  # Race categories
    )
    
    # Create a custom Multi-Task Dataset class
    class MultiTaskDataset(Dataset):
        def __init__(self, processed_dataset):
            self.processed_dataset = processed_dataset
            
        def __len__(self):
            return len(self.processed_dataset)
            
        def __getitem__(self, idx):
            item = self.processed_dataset[idx]
            pixel_values = torch.tensor(item['pixel_values'])
            age_label = item['label']
            race_label = item['race_label']
            
            return {
                'pixel_values': pixel_values, 
                'labels': age_label,
                'race_labels': race_label
            }
    
    # Convert processed datasets to multi-task datasets
    multi_task_train_dataset = MultiTaskDataset(processed_train)
    multi_task_test_dataset = MultiTaskDataset(processed_test)
    
    # Define data collator for multi-task learning
    def collate_fn(examples):
        # Convert to tensors
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        race_labels = torch.tensor([example["race_labels"] for example in examples])
        
        return {
            "pixel_values": pixel_values, 
            "labels": labels,
            "race_labels": race_labels
        }
    
    # Function to compute metrics
    def compute_metrics(eval_preds):
        age_logits, race_logits = eval_preds.predictions
        age_labels, race_labels = eval_preds.label_ids
        
        # Get predicted classes
        age_preds = np.argmax(age_logits, axis=1)
        race_preds = np.argmax(race_logits, axis=1)
        
        # Calculate metrics
        age_acc = accuracy_score(age_labels, age_preds)
        race_acc = accuracy_score(race_labels, race_preds)
        
        # Calculate precision, recall, F1 for age
        age_precision, age_recall, age_f1, _ = precision_recall_fscore_support(
            age_labels, age_preds, average='weighted', zero_division=0
        )
        
        # Calculate precision, recall, F1 for race
        race_precision, race_recall, race_f1, _ = precision_recall_fscore_support(
            race_labels, race_preds, average='weighted', zero_division=0
        )
        
        # Calculate combined accuracy
        combined_acc = (age_acc + race_acc) / 2
        
        # Try to save confusion matrices
        try:
            # Age confusion matrix
            age_cm = confusion_matrix(age_labels, age_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(age_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=AGE_GROUP_NAMES, yticklabels=AGE_GROUP_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Age Group Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'age_confusion_matrix.png'))
            plt.close()
            
            # Race confusion matrix
            race_cm = confusion_matrix(race_labels, race_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(race_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=RACE_NAMES, yticklabels=RACE_NAMES)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Race Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'race_confusion_matrix.png'))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save confusion matrices: {e}")
        
        # Return all metrics
        return {
            "accuracy": combined_acc,  # This is used for model selection
            "age_accuracy": age_acc,
            "age_precision": age_precision,
            "age_recall": age_recall,
            "age_f1": age_f1,
            "race_accuracy": race_acc,
            "race_precision": race_precision,
            "race_recall": race_recall,
            "race_f1": race_f1,
            "combined_accuracy": combined_acc
        }
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="mlflow",
        gradient_accumulation_steps=1 if torch.cuda.is_available() else 4,
        logging_steps=50,
        disable_tqdm=False,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),  # Mixed precision training for faster execution
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=multi_task_train_dataset,
        eval_dataset=multi_task_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # Start training
    print("Fine-tuning the model...")
    train_result = trainer.train()
    
    # Log metrics
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    
    # Evaluate model
    print("Evaluating the model...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['train_loss', 'accuracy', 'age_accuracy', 'age_f1', 'race_accuracy', 'race_f1'],
        'value': [
            train_metrics.get("train_loss", 0),
            eval_metrics.get("accuracy", 0),
            eval_metrics.get("age_accuracy", 0),
            eval_metrics.get("age_f1", 0),
            eval_metrics.get("race_accuracy", 0),
            eval_metrics.get("race_f1", 0)
        ]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Log MLflow metrics
    for key, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
    
    # Save model and processor
    print("Saving the fine-tuned model...")
    trainer.save_model(model_save_dir)
    image_processor.save_pretrained(model_save_dir)
    
    # Save model info
    with open(os.path.join(model_save_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Multi-task learning: Age and Race classification\n")
        f.write(f"Age group labels (4): {dict(zip(range(4), AGE_GROUP_NAMES))}\n")
        f.write(f"Race labels (5): {dict(zip(range(5), RACE_NAMES))}\n")
        f.write(f"Age accuracy: {eval_metrics.get('age_accuracy', 0):.4f}\n")
        f.write(f"Age F1 score: {eval_metrics.get('age_f1', 0):.4f}\n")
        f.write(f"Race accuracy: {eval_metrics.get('race_accuracy', 0):.4f}\n")
        f.write(f"Race F1 score: {eval_metrics.get('race_f1', 0):.4f}\n")
        f.write(f"Combined accuracy: {eval_metrics.get('combined_accuracy', 0):.4f}\n")
    
    print("Fine-tuning complete!")
    print(f"Model saved to: {model_save_dir}")
    print(f"Results saved to: {output_dir}")