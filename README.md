# 🚢 Ship Detection

> **VITMAV45 – Deep Learning in Practice with Python and LUA**  
> *Major Term Project – Airbus Ship Detection Challenge (Kaggle)*

---

## 📘 Project Overview

This repository contains our solution for the **first milestone** of the Deep Learning in Practice course.  
Our project is based on the [**Airbus Ship Detection Challenge**](https://www.kaggle.com/competitions/airbus-ship-detection) on Kaggle, which focuses on **detecting ships in satellite images** using **deep learning–based semantic segmentation**.

For **Milestone 1**, our objectives were:
1. **Data source identification** and creation of download scripts.  
2. **Data exploration and visualization** to understand image and label distributions.  
3. **Data preparation for model training**, resulting in **training, validation, and test datasets**.

For **Milestone 2**, our objectives were:
1. **Efficient data loading** using PyTorch Datasets and DataLoaders.  
2. **Model training** for ship segmentation.  
3. **Evaluation** using appropriate metrics.  
4. Reaching a stable level where only model improvement and final documentation remain.

---

## 👥 Team Information

Team name: 99_Problems_but_a_Batch_Aint_One

| Name | Neptun Code | Role |
|------|--------------|------|
| **Bologa Eduard** | DAM4AV | Documentation, GitHub repository integration, model evaluation visualizations, training workflow polishing |
| **Kozma Szabolcs András** | TKGQWN | GitHub repository setup, data preprocessing, efficient data loading, loss & metric functions |
| **Pünkösti Györk** | VCV3N5 | Kaggle integration, data visualization, model evaluation visualizations, UNet training |

---

## 🧠 Project Description

The goal of this project is to prepare, train and evaluate models on the **Airbus Ship Detection dataset**.  
The dataset includes **satellite images of varying sizes** with **RLE-encoded segmentation masks** marking the locations of ships.

Through **Milestone 1**, we:
- Collected and analyzed metadata from the Kaggle dataset.  
- Visualized image samples and segmentation masks.  
- Prepared structured data splits for training, validation, and testing.

---

## 🚀 Milestone 2 – Training & Evaluation

The notebook `notebooks/milestone2.ipynb` implements the second milestone:

- Efficient Data Loading
- Model Training
- Evaluation

---

## 📂 Repository Structure

| File / Folder                            | Description                                              |
| ---------------------------------------- | -------------------------------------------------------- |
| `.ipynb_checkpoints/`                    | Auto-generated Jupyter checkpoints.                      |
| `notebooks/`                             | Folder containing all Jupyter notebooks.                 |
| ├── `milestone1.ipynb`                   | Milestone 1 – data download, exploration, preprocessing. |
| ├── `milestone2.ipynb`                   | Milestone 2 – model training, evaluation, visualization. |
| ├── `ship_detection_work_notebook.ipynb` | Early exploratory notebook.                              |
| └── `ship_detection_work_notebook`       | Backup version without extension.                        |
| `.DS_Store`                              | macOS metadata file (ignore).                            |
| `.gitignore`                             | Specifies ignored files (datasets, cache, etc.).         |
| `LICENSE`                                | Project license (GPL-3.0).                               |
| `README.md`                              | Project documentation.                                   |
| `unet_ship_detection.pth`                | Saved U-Net model weights from Milestone 2.              |

---

## ▶️ How to Run

📌 Milestone 1 – Data Preparation

You can open the project directly in **Google Colab** by clicking the button below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/99ProblemsButABatchAint1/Ship_Detection/blob/main/notebooks/milestone1.ipynb)

Alternatively, you can clone the repository and run the notebook locally:

```bash
git clone https://github.com/99ProblemsButABatchAint1/Ship_Detection.git
cd Ship_Detection/notebooks
jupyter notebook milestone1.ipynb
```

📌 Milestone 2 – Training & Evaluation

Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/99ProblemsButABatchAint1/Ship_Detection/blob/main/notebooks/milestone2.ipynb)

Locally:

```bash
git clone https://github.com/99ProblemsButABatchAint1/Ship_Detection.git
cd Ship_Detection/notebooks
jupyter notebook milestone2.ipynb
```

## 📜 License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for details.

© 2025 – Team 99_Problems_but_a_Batch_Aint_One
Deep Learning in Practice with Python and LUA – Budapest University of Technology and Economics (BME)
