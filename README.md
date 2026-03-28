# CCD Optimization: LLM-Enriched Prompts & DINOv2 Label Propagation

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg)

This repository contains our optimization and enhancements for the paper **"Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification" (CVPR 2025)**. 

We extend the baseline Classifier-guided CLIP Distillation (CCD) framework with two novel contributions: LLM-enriched prompt embeddings and DINOv2 visual-similarity label propagation, achieving state-of-the-art performance on the PASCAL VOC 2007 dataset.

---

## 🚀 Key Contributions

### 1. LLM-Enriched Hybrid Prompt Embeddings
Standard zero-shot pipelines use generic prompts like `"a photo of a {class}"`. We utilize Large Language Models (ChatGPT/Claude) to generate rich, context-aware visual descriptions for each target class. 
* We extract embeddings for these descriptions using CLIP `RN50x64` and calculate the mean embedding for each class.
* We introduce an **α-blending parameter** to dynamically mix the standard CLIP prompts with our LLM-enriched embeddings. 
* **Result:** $\alpha=0.7$ yields the best performance, improving the baseline pseudo-label quality without requiring any additional training.

### 2. DINOv2 Visual-Similarity Label Propagation
To mitigate the noise inherent in CLIP-generated pseudo-labels, we incorporate semantic label smoothing based on visual similarity.
* We extract visual features for all training images using **DINOv2 (ViT-B/14)**.
* We construct a k-Nearest Neighbors (kNN) similarity graph ($k=10$) based on cosine similarity.
* At the beginning of training, we perform a weighted label propagation step to smooth pseudo-labels among visually similar images.

---

## 📊 Results (PASCAL VOC 2007)

Our implementation is evaluated using Mean Average Precision (mAP) over the 20 classes of PASCAL VOC 2007.

| Method | Best Test mAP | $\Delta$ vs. Baseline |
| :--- | :---: | :---: |
| **LLM Hybrid ($\alpha=0.7$)** | **90.99%** | **+0.17%** |
| Baseline (CCD) | 90.82% | — |
| DINOv2 Only | 90.76% | -0.06% |
| Combined (LLM + DINOv2) | 90.73% | -0.09% |

*The LLM Hybrid Prompt approach yields the best results. DINOv2 smoothing and the combined approach experienced slight interference with the iterative CAM refinement inherent in CCD.*

---

## 🛠️ Setup & Installation

This project is entirely self-contained within a Jupyter Notebook. All environment setups, dependency installations, and dataset/model downloads are handled programmatically within the notebook.

### Requirements
* Python 3.8+
* Jupyter Notebook or JupyterLab
* A CUDA-capable GPU is highly recommended.

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

---

## 💻 How to Run

1. Open `CCD_Project 2.ipynb` in your Jupyter environment.
2. Run the notebook cells sequentially from top to bottom.
3. The notebook naturally flows through the following phases:
   * **Part 1:** Environment setup (Installs requirements, downloads PASCAL VOC 2007, downloads CLIP model).
   * **Part 2:** Baseline execution.
   * **Part 3:** Generation of LLM-enriched embeddings and executing the $\alpha$-blending ablation study.
   * **Part 4 & 5:** DINOv2 feature extraction, graph construction, and model execution.
   * **Part 6:** Final results visualization generation (`final_results.png`, `llm_ablation.png`).

---

## 📁 Repository Structure

```text
.
├── CCD_Project 2.ipynb          # Main execution pipeline (Setup, experiments, visualizations)
├── CCD_Phase2_Report.html       # 2-column detailed technical report 
├── CCD_Phase2_Presentation.pptx # 4-slide project presentation
├── final_results.png            # Visualization of mAP over epochs
├── llm_ablation.png             # Visualization of the alpha-blending ablation study
└── README.md                    # This document
```
*(Note: Directories like `data/`, `pretrained/`, and `metadata/` will be generated dynamically when the notebook is run.)*

---

## 🎥 Demonstration

Watch our 15-minute video demonstration explaining the paper, our novel contributions, and a walkthrough of the codebase:

[**▶️ Watch on YouTube**](https://youtu.be/OP_m8KR2kWQ)

---

## 👥 Team

* **[Pratyush Chauhan]** - Base Paper + +DINOv2 Contribution 
* **[Aditya Jain]** -LLM Contribution Lead + Pipeline Lead
* **[Pratyush Chauhan]** - Experiments + Visualization + Documentation Lead
