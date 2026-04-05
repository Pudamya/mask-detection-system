# Face Mask Detection System 

A high-performance **Computer Vision application** that detects whether a person is wearing a face mask using a **custom-built Convolutional Neural Network (CNN)** and an interactive **Streamlit web interface**.

<p align="center">
  🔗 <a href="https://mask-detection-system.streamlit.app/">Live Application</a> •
  🎥 <a href="https://drive.google.com/file/d/1iaiqdjvcQes77XQOQAYuBOLt8aRU7hbd/view?usp=sharing">Demo Video</a>
</p>

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/d7737aee-bc05-45d8-a2ea-6d713a477883" width="45%" />
  <img width="45%" height="481" alt="image" src="https://github.com/user-attachments/assets/a8e9108f-af6e-4937-8565-1b452ff0fbb6" />
 />
</p>

---

## Project Overview

This project presents a complete **end-to-end deep learning pipeline** for face mask detection, combining image preprocessing, face detection, classification, and real-time deployment.

The system is designed to handle real-world variability such as:

* Close-up faces
* Blurred images
* Partial facial visibility
* Challenging lighting conditions

It integrates classical computer vision techniques with deep learning to achieve **robust and reliable predictions**.

---

## Key Features

* Custom-built CNN architecture (no transfer learning)
* High classification performance (~98–99% accuracy)
* Integrated face detection and validation pipeline
* Handles edge cases (blurred, partial, close-up faces)
* Smart fallback mechanism for difficult inputs
* Confidence-based prediction outputs
* Clean, modern, and user-friendly Streamlit interface

---

## Model Architecture

The system uses a **custom-designed Convolutional Neural Network (CNN)** tailored for binary image classification.

**Architecture highlights:**

* Input size: 128 × 128 RGB images
* Multiple convolutional layers with ReLU activation
* Max pooling layers for feature extraction
* Fully connected layers for classification
* Softmax output layer (2 classes: with_mask, without_mask)

This approach ensures full control over model design and optimization.

---

## Performance Evaluation

The model demonstrates strong performance on the test dataset:

* **Accuracy:** 98.85%
* **Precision:** 0.9886
* **Recall:** 0.9885
* **F1 Score:** 0.9885

The confusion matrix indicates:

* Very low misclassification rate
* Balanced performance across both classes
* Strong generalization capability

---

## System Workflow

### 1. Face Detection

* Uses OpenCV-based Haar Cascade classifiers
* Supports multiple detection strategies for improved robustness

### 2. Face Validation

* Ensures input contains a valid human face
* Filters out non-face images (e.g., QR codes, noise)

### 3. Image Preprocessing

* Resizing to 128 × 128
* Normalization and tensor conversion

### 4. Mask Classification

* Processed image is passed through the CNN
* Outputs class label with confidence score

### 5. Result Visualization

* Bounding boxes on detected faces
* Confidence score display
* Prediction summary

---

## User Interface (Streamlit)

The application provides an intuitive and interactive UI with:

* Image upload functionality
* Real-time inference
* Visual comparison (input vs prediction output)
* Confidence score visualization
* Prediction summaries and insights

---

## Project Structure

```
mask-detection-system/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── src/
│   ├── model.py               # Custom CNN model
│   ├── inference.py           # Prediction logic
│   ├── preprocessing.py       # Image preprocessing
│   └── main.py                # Training & evaluation
│
├── models/
│   └── best_model.pth         # Trained model
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/Pudamya/mask-detection-system
cd mask-detection-system
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app/streamlit_app.py
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud** for easy accessibility and real-time interaction.

🔗 [**Live Application**](https://mask-detection-system.streamlit.app/)
🎥 [**Demo Video**](https://drive.google.com/file/d/1iaiqdjvcQes77XQOQAYuBOLt8aRU7hbd/view?usp=sharing)

---

## Author

**Pudamya Rathnayake**
BSc (Hons) Artificial Intelligence & Data Science

🔗 LinkedIn: https://www.linkedin.com/in/pudamya-rathnayake
🔗 GitHub: https://github.com/Pudamya

