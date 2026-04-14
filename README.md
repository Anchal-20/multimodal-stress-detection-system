# 🧠 Multimodal Stress Detection System

🚀 **Published Patent Project | Computer Vision + Physiological Signal Processing + AI**

---

## 📌 Overview

This project presents a **multimodal stress detection system** that combines:

- 🧑‍💻 Facial Emotion Recognition (Computer Vision)
- ❤️ Physiological Signals (SpO2, Pulse Rate, Temperature)
- 🤖 Machine Learning (CNN + K-Means Clustering)

The system performs **real-time stress detection** using webcam input and sensor-based data, enabling **accurate and robust stress assessment** compared to single-modality approaches.

---

## 🏆 Patent

📄 **Stress Monitoring Device using Physiological and Psychological Parameters**  
📍 Published in Indian Patent Office (2025)  
📌 Application No: 202521066859A  

This project is based on a **published patent**, where a multimodal system is designed to:
- Analyze real-time physiological and psychological signals  
- Establish personalized stress baselines  
- Provide feedback via integrated system  

---

## 🧠 Key Features

- 🎥 Real-time stress detection using webcam (OpenCV)
- 🧠 CNN-based facial emotion recognition model
- 📊 K-Means clustering on physiological signals
- 📉 PCA-based visualization for stress patterns
- 🔄 Multimodal fusion for improved accuracy
- 💾 Model persistence using HDF5 and Joblib

---

## 🏗️ System Architecture

### Input Layer:
- Face Image (Webcam)
- Physiological Signals (Temperature, SpO2, Pulse Rate)

### Processing:
- CNN → Emotion → Stress Level
- K-Means → Cluster → Stress Pattern

### Output:
- Stress Level Classification:
  - 🟢 Low Stress  
  - 🟡 Medium Stress  
  - 🔴 High Stress  

---

## 🧪 Models Used

### 🔹 1. CNN Model (Emotion → Stress)
- Input: 48×48 grayscale facial images  
- Architecture:
  - Conv2D (32) → MaxPool  
  - Conv2D (64) → MaxPool  
  - Conv2D (128) → MaxPool  
  - Dense (256) + Dropout  
- Output: 3 stress levels  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  

---

### 🔹 2. K-Means Clustering (Physiological Data)
- Features:
  - Temperature  
  - SpO2  
  - Pulse Rate  

- Techniques used:
  - Feature Scaling (StandardScaler)
  - Elbow Method
  - Silhouette Score
  - PCA for visualization
