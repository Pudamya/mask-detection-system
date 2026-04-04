import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


class BasicInference:
    """
    Two-stage pipeline:
    Stage 1 — Face Detection: Haar Cascade finds WHERE faces are in image
    Stage 2 — Classification: Our CNN decides mask/no-mask for each face
    """

    def __init__(self, model, device, img_size=128, classes=None):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.classes = classes or ['with_mask', 'without_mask']

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def is_blurry(self, image, threshold=80.0):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold, variance

    def is_too_small(self, w, h, min_size=35):
        return w < min_size or h < min_size

    def format_prediction_label(self, predicted_class, confidence, threshold=55.0):
        if confidence < threshold:
            return "uncertain"
        return predicted_class

    def classify_face(self, input_data):
        self.model.eval()

        if isinstance(input_data, Image.Image):
            tensor = self.transform(input_data).unsqueeze(0).to(self.device)
        else:
            tensor = input_data.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(probabilities))
            predicted_class = self.classes[predicted_idx]
            confidence = float(probabilities[predicted_idx] * 100)

        return predicted_class, confidence, probabilities

    def classify_face_tta(self, pil_face):
        variants = [
            pil_face,
            pil_face.transpose(Image.FLIP_LEFT_RIGHT)
        ]

        probs_list = []

        self.model.eval()
        with torch.no_grad():
            for img in variants:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                probs_list.append(probs)

        mean_probs = np.mean(probs_list, axis=0)
        predicted_idx = int(np.argmax(mean_probs))
        predicted_class = self.classes[predicted_idx]
        confidence = float(mean_probs[predicted_idx] * 100)

        return predicted_class, confidence, mean_probs

    def _apply_nms(self, faces, overlap_threshold=0.3):
        boxes = []
        for (x, y, w, h) in faces:
            if self.is_too_small(w, h, min_size=35):
                continue
            boxes.append([x, y, x + w, y + h])

        if len(boxes) == 0:
            return []

        boxes = np.array(boxes, dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, ix2 - ix1)
            inter_h = np.maximum(0, iy2 - iy1)
            inter = inter_w * inter_h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= overlap_threshold)[0]
            order = order[inds + 1]

        result = []
        for i in keep:
            b = boxes[i]
            result.append((
                int(b[0]),
                int(b[1]),
                int(b[2] - b[0]),
                int(b[3] - b[1])
            ))
        return result

    def detect_images(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(45, 45),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.03,
                minNeighbors=4,
                minSize=(35, 35),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        if len(faces) > 0:
            faces = self._apply_nms(faces, overlap_threshold=0.3)

        results = []
        annotated = image_rgb.copy()

        if len(faces) == 0:
            h_img, w_img = image_rgb.shape[:2]

            pil_img = Image.fromarray(image_rgb)
            predicted_class, confidence, probabilities = self.classify_face(pil_img)
            display_class = self.format_prediction_label(predicted_class, confidence, threshold=55.0)

            results.append({
                'bbox': (0, 0, w_img, h_img),
                'class': display_class,
                'raw_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'blur_detected': False,
                'blur_score': 0.0,
                'used_full_image_fallback': True
            })

            if display_class == 'with_mask':
                color = (0, 180, 0)
            elif display_class == 'without_mask':
                color = (220, 40, 40)
            else:
                color = (255, 165, 0)

            label = f"{display_class} ({confidence:.1f}%)"
            cv2.rectangle(annotated, (10, 10), (w_img - 10, h_img - 10), color, 3)
            cv2.putText(
                annotated,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            return annotated, results

        for (x, y, w, h) in faces:
            if self.is_too_small(w, h, min_size=35):
                continue

            pad_w = int(0.18 * w)
            pad_h = int(0.18 * h)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image_rgb.shape[1], x + w + pad_w)
            y2 = min(image_rgb.shape[0], y + h + pad_h)

            face_img = image_rgb[y1:y2, x1:x2]
            is_blur, blur_score = self.is_blurry(face_img)
            pil_face = Image.fromarray(face_img)

            predicted_class, confidence, probabilities = self.classify_face_tta(pil_face)
            display_class = self.format_prediction_label(predicted_class, confidence, threshold=55.0)

            results.append({
                'bbox': (x, y, w, h),
                'class': display_class,
                'raw_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'blur_detected': is_blur,
                'blur_score': float(blur_score),
                'used_full_image_fallback': False
            })

            if display_class == 'with_mask':
                color = (0, 180, 0)
            elif display_class == 'without_mask':
                color = (220, 40, 40)
            else:
                color = (255, 165, 0)

            label = f"{display_class} ({confidence:.1f}%)"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y1 = max(0, y - th - 10)
            cv2.rectangle(annotated, (x, text_y1), (x + tw + 4, y), color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 2, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated, results

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

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print("\nTest Set Evaluation")
        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.classes))

        os.makedirs(save_dir, exist_ok=True)
        metrics = {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1)
        }

        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        cm = confusion_matrix(all_labels, all_preds)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(np.arange(len(self.classes)))
        ax.set_yticks(np.arange(len(self.classes)))
        ax.set_xticklabels(self.classes)
        ax.set_yticklabels(self.classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved.")

        return all_preds, all_labels