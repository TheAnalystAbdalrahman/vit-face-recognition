#!/usr/bin/env python3
# Utility functions for ViT face recognition project

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import random
from sklearn.metrics import confusion_matrix
import cv2

# Age group names for consistent labeling
AGE_GROUP_NAMES = ['Child/Youth (0-17)', 'Young Adult (18-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
RACE_NAMES = ['White', 'Black', 'Asian', 'Indian', 'Others']

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_project_dirs():
    """Create necessary project directories"""
    dirs = [
        "./data",
        "./fine_tuned_model",
        "./results",
        "./real_world_data/test_images",
        "./evaluation_results", 
        "./real_world_results"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Project directories created.")

def map_age_to_group(age_idx, dataset):
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

def visualize_dataset_samples(dataset, num_samples=5):
    """Visualize random samples from the dataset"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        example = dataset[idx]
        image = example['image']
        age_idx = example['age']
        gender_idx = example['gender']
        race_idx = example['race']
        
        # Get feature names
        age_name = dataset.features['age'].names[age_idx]
        gender_name = dataset.features['gender'].names[gender_idx]
        race_name = dataset.features['race'].names[race_idx]
        
        # Age group
        age_group = map_age_to_group(age_idx, dataset.dataset)
        age_group_name = AGE_GROUP_NAMES[age_group]
        
        # Display image and metadata
        axes[i].imshow(image)
        axes[i].set_title(f"Age: {age_name}\nGroup: {age_group_name}\nGender: {gender_name}\nRace: {race_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, classes=AGE_GROUP_NAMES, normalize=False, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def detect_faces(image_path):
    """Detect faces in an image using OpenCV's Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Convert to RGB for display
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Return detected faces as bounding boxes
    return rgb_img, faces

def visualize_face_detection(image_path, output_path=None):
    """Visualize face detection on an image"""
    img, faces = detect_faces(image_path)
    
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Convert back to PIL for display
    pil_img = Image.fromarray(img)
    
    # Save if output path provided
    if output_path:
        pil_img.save(output_path)
    
    return pil_img, len(faces)

def get_dataset_statistics(dataset):
    """Get basic statistics about the dataset"""
    stats = {}
    
    # Number of examples
    stats['total_examples'] = len(dataset)
    
    # Class distributions
    age_counts = {}
    gender_counts = {}
    race_counts = {}
    age_group_counts = {name: 0 for name in AGE_GROUP_NAMES}
    
    for example in dataset:
        # Age counts
        age_idx = example['age']
        age_name = dataset.features['age'].names[age_idx]
        if age_name in age_counts:
            age_counts[age_name] += 1
        else:
            age_counts[age_name] = 1
        
        # Age group counts
        age_group = map_age_to_group(age_idx, dataset.dataset)
        age_group_name = AGE_GROUP_NAMES[age_group]
        age_group_counts[age_group_name] += 1
        
        # Gender counts
        gender_idx = example['gender']
        gender_name = dataset.features['gender'].names[gender_idx]
        if gender_name in gender_counts:
            gender_counts[gender_name] += 1
        else:
            gender_counts[gender_name] = 1
        
        # Race counts
        race_idx = example['race']
        race_name = dataset.features['race'].names[race_idx]
        if race_name in race_counts:
            race_counts[race_name] += 1
        else:
            race_counts[race_name] = 1
    
    stats['age_counts'] = age_counts
    stats['age_group_counts'] = age_group_counts
    stats['gender_counts'] = gender_counts
    stats['race_counts'] = race_counts
    
    return stats

def prepare_exploratory_plots(dataset, output_dir='./results'):
    """Create exploratory plots for the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get statistics
    stats = get_dataset_statistics(dataset)
    
    # Plot age distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(stats['age_counts'].keys()), y=list(stats['age_counts'].values()))
    plt.title('Age Distribution')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    
    # Plot age group distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(stats['age_group_counts'].keys()), y=list(stats['age_group_counts'].values()))
    plt.title('Age Group Distribution')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_group_distribution.png'))
    
    # Plot gender distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(stats['gender_counts'].keys()), y=list(stats['gender_counts'].values()))
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    
    # Plot race distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(stats['race_counts'].keys()), y=list(stats['race_counts'].values()))
    plt.title('Race Distribution')
    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_distribution.png'))
    
    return stats

if __name__ == "__main__":
    # Test utility functions
    create_project_dirs()
    print("Utilities loaded successfully.")