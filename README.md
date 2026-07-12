# 🖥️ Smart PCB Defect Detection and Classification

An AI-powered web application that automatically detects and classifies defects in Printed Circuit Boards (PCBs) using Deep Learning and Computer Vision techniques.

## 🚀 Project Overview

Manual PCB inspection is time-consuming and prone to human error. This project automates the inspection process by using a trained deep learning model to identify PCB defects from uploaded images.

The application provides an easy-to-use web interface where users can upload PCB images and receive instant predictions.

---

## ✨ Features

- Detects PCB defects automatically
- Classifies multiple defect categories
- Fast image inference
- User-friendly web interface
- Deep Learning based prediction
- Supports image upload
- Real-time prediction results

---

## 🧠 Defect Classes

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Deep Learning

- PyTorch

### Computer Vision

- OpenCV

### Web Framework

- Flask

### Libraries

- Torch
- Torchvision
- NumPy
- Pillow
- OpenCV
- JSON

---

## 📂 Project Structure

```
Smart-PCB-defect-detection-and-classification
│
├── app.py
├── api.py
├── inference_backend.py
├── class_names.json
├── best_model.pth
├── sample/
├── Template/
├── static/
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/kanhumishra/Smart-PCB-defect-detection-and-classification.git
```

Move into the project

```bash
cd Smart-PCB-defect-detection-and-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

The application will start on

```
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

1. Upload PCB Image
2. Image Preprocessing
3. Deep Learning Model Prediction
4. Display Defect Class
5. Show Prediction Result

---

## 🎯 Model

The project uses a trained PyTorch model for PCB defect classification.

Model File:

```
best_model.pth
```

---

## 📈 Future Improvements

- Confidence Score
- Grad-CAM Visualization
- Multiple Image Upload
- Defect Localization
- Database Integration
- REST API Deployment
- Docker Support

---

## 👨‍💻 Author

**Kanhu Charan Mishra**

MCA Student | Data Analyst | AI & Machine Learning Enthusiast

GitHub:
https://github.com/kanhumishra

LinkedIn:
(Add your LinkedIn URL)

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
