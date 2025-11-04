# 🚢 Ship Detection

> **VITMAV45 – Deep Learning a gyakorlatban Python és LUA alapon**  
> *Nagy házi feladat projekt – Airbus Ship Detection Challenge (Kaggle)*

---

## 📘 Project Overview

This repository contains our solution for the **first milestone** of the Deep Learning in Practice course.  
Our project is based on the [**Airbus Ship Detection Challenge**](https://www.kaggle.com/competitions/airbus-ship-detection) on Kaggle, which focuses on **detecting ships in satellite images** using **deep learning–based semantic segmentation**.

For this milestone, our objectives are:
1. **Data source identification** and creation of download scripts.
2. **Data exploration and visualization** to understand image and label distributions.
3. **Data preparation for model training**, resulting in **training, validation, and test datasets**.

---

## 👥 Team Information

Team name: 99_Problems_but_a_Batch_Aint_One

| Name | Neptun Code | Role |
|------|--------------|------|
| **Bologa Eduard** | DAM4AV | Documentation, GitHub repository integration |
| **Kozma Szabolcs András** | TKGQWN | GitHub repository setup, data preprocessing|
| **Pünkösti Györk** | VCV3N5 | Kaggle integration, data visualization |

---

## 🧠 Project Description

The goal of this project is to prepare and explore the **Airbus Ship Detection dataset** for subsequent model training.  
The dataset includes **satellite images of varying sizes** with **RLE-encoded segmentation masks** marking the locations of ships.

Through this milestone, we:
- Collected and analyzed metadata from the Kaggle dataset.  
- Visualized image samples and segmentation masks.  
- Prepared cleaned and structured data splits for model training, validation, and testing.

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| `ship_detection_work_notebook.ipynb` | Main Jupyter Notebook containing the data download, exploration, and preprocessing steps. |
| `ship_detection_work_notebook` | Backup or alternate version of the notebook. |
| `README.md` | Project documentation (this file). |
| `LICENSE` | Project license (GPL-3.0). |

---

## ▶️ How to Run

You can open the project directly in **Google Colab** by clicking the button below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/99ProblemsButABatchAint1/Ship_Detection/blob/main/ship_detection_work_notebook.ipynb)

Alternatively, you can clone the repository and run the notebook locally:

```bash
git clone https://github.com/99ProblemsButABatchAint1/Ship_Detection.git
cd Ship_Detection
jupyter notebook ship_detection_work_notebook.ipynb
