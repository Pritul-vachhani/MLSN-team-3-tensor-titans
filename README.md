# MLSN-team-3-tensor-titans
# 🤟 ASL Hand Sign Recognition (A–E)

This project implements a real-time American Sign Language (ASL) hand sign recognition system for the letters **A** to **E** using deep learning and computer vision. It uses a trained CNN model to classify ASL hand gestures from a webcam feed, powered by TensorFlow, OpenCV, and MediaPipe.

---

## 🚀 Project Structure

- `asl_model.h5` – Trained TensorFlow model for classifying ASL letters A–E.
- `asl_demo.py` – Live webcam-based ASL detection and classification script.
- `asl_dataset/` – Dataset folder with training images of ASL hand signs.
- `training_script.py` – Script used to train and validate the CNN model.
- `requirements.txt` – Python dependencies.

---

## 🧠 Model Training

- **Dataset**: ASL Alphabet dataset (A–E subset).
- **Image Size**: 224x224 for MobileNet compatibility.
- **Preprocessing**:
  - Rescaling and normalization.
  - Data augmentation (rotation, zoom, shifts, horizontal flip).
- **Model**: Transfer learning using MobileNetV2 + custom dense layers.
- **Evaluation**: Achieved high accuracy on validation data.

---

## 🖥️ Running the Live Demo

### Requirements
```bash
pip install -r requirements.txt
