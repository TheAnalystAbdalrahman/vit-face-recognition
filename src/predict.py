import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from glob import glob
import cv2
import argparse
from tqdm import tqdm
import seaborn as sns
from safetensors.torch import load_file as safe_load


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define class names
AGE_GROUP_NAMES = ['Child/Youth (0-17)', 'Young Adult (18-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
RACE_NAMES = ['White', 'Black', 'Asian', 'Indian', 'Others']

# Define the same model architecture used for training
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

# Create output directory
output_dir = "./real_world_results"
os.makedirs(output_dir, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test face recognition model on real-world images')
parser.add_argument('--image_dir', type=str, default='./real_world_data/test_images',
                    help='Directory containing test images')
parser.add_argument('--model_dir', type=str, default='./fine_tuned_model',
                    help='Directory containing fine-tuned model')
args = parser.parse_args()

def preprocess_image(image_path):
    """Preprocess a single image for inference"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return image, transform(image)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def detect_and_crop_faces(image_path, save_crops=False):
    """Detect faces in image and return cropped faces"""
    # Load pre-trained face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If no faces detected, return original image
    if len(faces) == 0:
        print(f"No faces detected in {image_path}, using full image.")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        return [pil_img]
    
    # Process each detected face
    cropped_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        # Add some margin around the face
        margin = int(0.3 * max(w, h))
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(img.shape[1], x + w + margin)
        y_end = min(img.shape[0], y + h + margin)
        
        # Crop the face
        face_img = img[y_start:y_end, x_start:x_end]
        
        # Convert to RGB (PIL format)
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)
        cropped_faces.append(pil_face)
        
        # Save cropped face if requested
        if save_crops:
            base_name = os.path.basename(image_path)
            crop_path = os.path.join(output_dir, f"crop_{base_name.split('.')[0]}_{i}.jpg")
            pil_face.save(crop_path)
    
    return cropped_faces

def predict_age_and_race(image_path, model, processor):
    """Predict age group and race for image"""
    # Try to detect and crop faces
    faces = detect_and_crop_faces(image_path, save_crops=True)
    
    if not faces:
        return None, None, None
    
    results = []
    for i, face in enumerate(faces):
        # Preprocess face
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        processed_face = transform(face)
        
        if processed_face is None:
            continue
        
        # Make prediction
        inputs = processor(images=face, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get age predictions
        age_logits = outputs['age_logits']
        age_probabilities = torch.nn.functional.softmax(age_logits, dim=1)
        age_class = torch.argmax(age_probabilities, dim=1).cpu().numpy()[0]
        age_confidence = age_probabilities[0, age_class].cpu().numpy().item()
        
        # Get race predictions
        race_logits = outputs['race_logits']
        race_probabilities = torch.nn.functional.softmax(race_logits, dim=1)
        race_class = torch.argmax(race_probabilities, dim=1).cpu().numpy()[0]
        race_confidence = race_probabilities[0, race_class].cpu().numpy().item()
        
        # Store results for this face
        results.append({
            'face_index': i,
            'predicted_age_class': age_class,
            'age_confidence': age_confidence,
            'predicted_age_group': AGE_GROUP_NAMES[age_class],
            'predicted_race_class': race_class,
            'race_confidence': race_confidence,
            'predicted_race': RACE_NAMES[race_class]
        })
    
    return faces, results, image_path

def process_images(image_dir, model_dir):
    """Process all images in directory"""
    # Load processor
    processor = ViTImageProcessor.from_pretrained(model_dir)
    
    # Load the base ViT model
    base_model_name = "google/vit-base-patch16-224"
    
    # Create and load our custom model
    model = ViTForMultiTaskClassification(base_model_name)
    
    # Load the state dict
    state_dict = safe_load(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob(os.path.join(image_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Processing {len(image_paths)} images...")
    
    all_results = []
    for image_path in tqdm(image_paths):
        faces, results, img_path = predict_age_and_race(image_path, model, processor)
        
        if results:
            # Add image path to results
            for result in results:
                result['image_path'] = os.path.basename(img_path)
            
            all_results.extend(results)
            
            # Create visualization
            fig, axes = plt.subplots(1, len(faces), figsize=(5*len(faces), 5))
            if len(faces) == 1:
                axes = [axes]  # Make it iterable
                
            for i, (face, result) in enumerate(zip(faces, results)):
                axes[i].imshow(face)
                axes[i].set_title(f"Age: {result['predicted_age_group']}\nAge Conf: {result['age_confidence']:.2f}\n" + 
                                 f"Race: {result['predicted_race']}\nRace Conf: {result['race_confidence']:.2f}")
                axes[i].axis('off')
            
            plt.tight_layout()
            base_name = os.path.basename(img_path).split('.')[0]
            plt.savefig(os.path.join(output_dir, f"prediction_{base_name}.png"))
            plt.close(fig)
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'real_world_predictions.csv'), index=False)
        
        # Summary visualization for age groups
        plt.figure(figsize=(10, 6))
        sns.countplot(x='predicted_age_group', data=results_df)
        plt.title('Predicted Age Groups')
        plt.xlabel('Age Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predicted_age_distribution.png'))
        
        # Summary visualization for race
        plt.figure(figsize=(10, 6))
        sns.countplot(x='predicted_race', data=results_df)
        plt.title('Predicted Race')
        plt.xlabel('Race')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predicted_race_distribution.png'))
        
        # Confidence by age group
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='predicted_age_group', y='age_confidence', data=results_df)
        plt.title('Age Prediction Confidence by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Confidence')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_confidence_by_class.png'))
        
        # Confidence by race
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='predicted_race', y='race_confidence', data=results_df)
        plt.title('Race Prediction Confidence by Race')
        plt.xlabel('Race')
        plt.ylabel('Confidence')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'race_confidence_by_class.png'))
        
        print(f"Results saved to {output_dir}")
        return results_df
    else:
        print("No valid predictions were made.")
        return None

if __name__ == "__main__":
    results = process_images(args.image_dir, args.model_dir)