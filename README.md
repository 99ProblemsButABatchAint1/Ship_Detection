# 🚢 Ship Detection (Final Milestone)

### VITMAV45 – Deep Learning in Practice with Python and LUA
**Major Term Project – Airbus Ship Detection Challenge (Kaggle)**

### **Megajánlott jegyért**

---

## 📘 Project Overview
This repository contains our final solution for the Deep Learning in Practice course.
Our project tackles the **Airbus Ship Detection Challenge** on Kaggle, focusing on detecting ships in satellite images using a robust **Two-Stage Deep Learning Pipeline**.

In addition to the research notebooks, we have deployed the model as a **Full-Stack Web Application** allowing users to upload satellite images and receive real-time detection overlays.

---

## 🎯 Key Objectives & Achievements

* **Milestone 1:** Data exploration, visualization, and preparation of train/val/test splits.
* **Milestone 2:** Initial U-Net training, data loading optimization, and baseline evaluation.
* **Milestone 3 (Final):**
    * **Advanced Architecture:** Implemented a Two-Stage Pipeline (Classifier + Segmenter) to handle class imbalance.
    * **Optimization:** Reduced False Positives significantly by filtering empty images.
    * **Performance:** Achieved high F2 Score and IoU on the unseen test set.
    * **Web Deployment:** Containerized the solution using Docker, FastAPI, and React.

---

## 👥 Team Information
**Team Name:** 99_Problems_but_a_Batch_Aint_One

| Name | Neptun Code | Role |
| :--- | :--- | :--- |
| **Bologa Eduard** | DAM4AV | Documentation, GitHub integration, pipeline architecture design, final report |
| **Kozma Szabolcs András** | TKGQWN | Data loading optimization, loss function tuning (Tversky), metric implementation |
| **Pünkösti Györk** | VCV3N5 | Model training (Classifier & Segmenter), inference pipeline, visualizations, Dockerization |

---

## 🧠 The Solution: Two-Stage Pipeline

Due to the extreme class imbalance (most satellite images contain only sea), a single segmentation model often produces false positives ("ghost ships") on waves or clouds. We solved this with a two-step approach:

### 1. Stage 1: The Classifier (Filter)
* **Model:** ResNet34 (Binary Classifier)
* **Input:** 384x384 resized image.
* **Goal:** Rapidly determine if an image contains *any* ship.
* **Result:** Filters out **~70%** of empty images before they reach the heavier segmentation model.

### 2. Stage 2: The Segmenter (Pixel-Level Detection)
* **Model:** U-Net with EfficientNet-B4 Encoder.
* **Input:** 768x768 full-resolution image (only if Stage 1 detects a ship).
* **Loss Function:** Tversky Loss ($\alpha=0.3$, $\beta=0.7$) to prioritize Recall (F2 Score).
* **Goal:** Precise pixel-wise segmentation of ship boundaries.

---

## 🚀 Optimization & Results

We implemented several key optimizations to improve performance and training speed:

* **Fast Data Loading:** Pre-processed RLE strings into list objects before the Dataset loop to remove CPU bottlenecks.
* **Threshold Tuning:** Automated script to find the optimal probability threshold (e.g., 0.35 instead of 0.5) that maximizes the F2 Score.
* **Tversky Loss:** Switched from Dice Loss to Tversky Loss to penalize False Negatives more heavily (crucial for the F2 metric).

### 📊 Final Performance (Test Set)

| Metric | Score | Description |
| :--- | :--- | :--- |
| **F2 Score** | **0.8977** | (Primary Metric) Harmonic mean of precision and recall, weighing recall higher. |
| **Precision** | 81.58% | Percentage of predicted ships that are actually ships. |
| **Recall** | 92.08% | Percentage of real ships that were successfully detected. |

---

## 💻 How to Run the Web App

We have containerized the application using **Docker** for easy deployment.

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* **Model Weights:** Ensure you have the trained model files (`best_classifier.pth` and `best_ship_segmenter.pth`).

### Quick Start
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
    cd YOUR_REPO
    ```

2.  **Add Model Weights:**
    * *Note: Large model files are ignored by git.*
    * Manually copy your `.pth` files into the `backend/models/` directory.

3.  **Run with Docker Compose:**
    ```bash
    docker compose up --build
    ```
    *(Note: If you run into network errors on University WiFi/VPN, try disconnecting the VPN or setting the MTU in Docker settings to 1300).*

4.  **Access the App:**
    * Frontend (React): [http://localhost:5173](http://localhost:5173)
    * Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Repository Structure

```text
├── LICENSE
├── notebooks
│   ├── milestone1.ipynb
│   ├── milestone2.ipynb
│   ├── milestone.ipynb
│   ├── notebook.ipynb
│   ├── ship_detection_work_notebook
│   ├── ship_detection_work_notebook.ipynb
│   └── unet_ship_detection.pth
├── project
│   ├── backend
│   │   ├── app
│   │   │   ├── deps.py
│   │   │   ├── main.py
│   │   │   ├── __pycache__
│   │   │   │   ├── deps.cpython-313.pyc
│   │   │   │   └── main.cpython-313.pyc
│   │   │   ├── routes
│   │   │   │   ├── detect.py
│   │   │   │   ├── health.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── detect.cpython-313.pyc
│   │   │   │       └── health.cpython-313.pyc
│   │   │   └── services
│   │   │       ├── detector.py
│   │   │       ├── detector_stub.py
│   │   │       └── __pycache__
│   │   │           ├── detector.cpython-313.pyc
│   │   │           └── detector_stub.cpython-313.pyc
│   │   ├── Dockerfile
│   │   ├── models
│   │   │   ├── best_classifier.pth
│   │   │   └── best_ship_segmenter.pth
│   │   ├── __pycache__
│   │   └── requirements
│   ├── docker-compose.yml
│   ├── Dockerfile.dev
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   │   └── vite.svg
├── README.md
├── final_milestone_documentation.pdf
├── src
│   ├── best_classifier.pth
│   ├── best_ship_segmenter.pth
│   ├── dataset.py
│   ├── inference.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── __pycache__
│   │   ├── dataset.cpython-313.pyc
│   │   ├── losses.cpython-313.pyc
│   │   ├── metrics.cpython-313.pyc
│   │   ├── model.cpython-313.pyc
│   │   ├── rle.cpython-313.pyc
│   │   └── transforms.cpython-313.pyc
│   ├── rle.py
│   ├── train.py
│   ├── transforms.py
│   └── tune_thresholds.py
├── unet_ship_detection.pth
└── weights
    ├── best_classifier.pth
    └── best_ship_segmenter.pth

```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/99ProblemsButABatchAint1/Ship_Detection/blob/main/notebooks/milestone.ipynb)

