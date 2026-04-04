import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import seaborn as sns
import os

# Handles face detection + mask classification on static images and evaluates the model on the test set.
class BasicInference:
    """
    Two-stage pipeline:
    Stage 1 — Face Detection: Haar Cascade finds WHERE faces are in image
    Stage 2 — Classification: Our CNN decides mask/no-mask for each face
    """

    def __init__(self, model, device, img_size=128,
                 classes=None):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.classes = classes or ['with_mask', 'without_mask']

        # Load OpenCV's pre-trained face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Preprocessing for inference
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect_images(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # scaleFactor: how much image size is reduced per scale
        # minNeighbors: how many neighbors a rectangle must retain
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        results = []
        annotated = image_rgb.copy()

        if len(faces) == 0:
            print("No faces detected in the image.")
            return annotated, results

        for (x, y, w, h) in faces:
            # Crop face region
            face_img = image_rgb[y:y+h, x:x+w]
            pil_face = Image.fromarray(face_img)

            # Preprocess + classify
            tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
            predicted_class, confidence, probabilities = self.classify_face(tensor)

            results.append({
                'bbox': (x, y, w, h),
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            })

            # Draw bounding box: green = mask, red = no mask
            color = (0, 200, 0) if predicted_class == 'with_mask' else (200, 0, 0)
            label = f"{predicted_class} ({confidence:.1f}%)"
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated, results

    # Run a single face tensor through the model and return prediction.
    def classify_face(self, tensor):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.classes[predicted_idx]
            confidence = probabilities[predicted_idx] * 100

        return predicted_class, confidence, probabilities

    def evaluate_on_test_set(self, test_loader, save_dir='results'):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Print report
        print("\nTest Set Evaluation")
        print(f"Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
        print(f"F1 Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                     target_names=self.classes))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.show()
        print("Confusion matrix saved.")

        return all_preds, all_labels