import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

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

        # Equalize histogram — improves detection in bright/dark images
        gray = cv2.equalizeHist(gray)

        # Detect faces with stricter parameters to avoid duplicates
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # smaller step = more thorough but slower
            minNeighbors=8,       # higher = fewer false positives / duplicates
            minSize=(80, 80),     # ignore tiny detections
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # ── Non-Maximum Suppression ──────────────────────────────────
        # CONCEPT: Even with strict params, sometimes overlapping boxes
        # appear for the same face. NMS keeps only the best box when
        # two boxes overlap more than the threshold (50% overlap here).
        if len(faces) > 0:
            faces = self._apply_nms(faces, overlap_threshold=0.3)

        results = []
        annotated = image_rgb.copy()

        if len(faces) == 0:
            print("No faces detected in the image.")
            return annotated, results

        for (x, y, w, h) in faces:
            # Add small padding around face for better classification
            pad = int(0.1 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image_rgb.shape[1], x + w + pad)
            y2 = min(image_rgb.shape[0], y + h + pad)

            face_img = image_rgb[y1:y2, x1:x2]
            pil_face = Image.fromarray(face_img)

            tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
            predicted_class, confidence, probabilities = self.classify_face(tensor)

            results.append({
                'bbox': (x, y, w, h),
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            })

            # Green = mask, Red = no mask
            color = (0, 200, 0) if predicted_class == 'with_mask' else (200, 0, 0)
            label = f"{predicted_class} ({confidence:.1f}%)"

            # Draw filled rectangle behind text for readability
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated, results


    def _apply_nms(self, faces, overlap_threshold=0.3):
        # Non-Maximum Suppression - removes duplicate overlapping face boxes.
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes, dtype=np.float32)

        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]  # process largest boxes first

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute intersection with all remaining boxes
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, ix2 - ix1)
            inter_h = np.maximum(0, iy2 - iy1)
            inter   = inter_w * inter_h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with low overlap only
            inds = np.where(iou <= overlap_threshold)[0]
            order = order[inds + 1]

        # Convert back to (x, y, w, h)
        result = []
        for i in keep:
            b = boxes[i]
            result.append((int(b[0]), int(b[1]),
                        int(b[2] - b[0]), int(b[3] - b[1])))
        return result

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