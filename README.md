# Multi-Task Cardiac Cine-MRI Analysis via Topology-Preserving Multi-Class Loss

This repository contains the complete implementation for the MTech thesis research on joint multi-task cardiac MRI analysis. The framework uses a shared 2D convolutional encoder to simultaneously perform 4-class anatomical segmentation (Background, LV, RV, MYO) and 5-class clinical cardiomyopathy risk classification (Normal, MINF, DCM, HCM, ARV) on short-axis cine-MRI slices. 

To prevent non-manifold shapes (such as broken myocardial rings or floating pixel islands) without incurring heavy 3D convolution or persistent homology computational costs, the network is regularized using a differentiable **Directional Gradient Boundary Penalty (\\(L_{topo}\\))**.

---

## 📌 Project Brief & Research Context

Standard 2D deep learning models trained on volumetric MRI slices suffer from a lack of global spatial context. This results in physically impossible anatomical predictions—such as overlapping ventricular cavities or fragmented myocardial walls—that cardiologists would immediately reject. While 3D networks can preserve volumetric continuity, they introduce an \\(O(N^3)\\) computational tax, leading to VRAM bottlenecks and tiny batch sizes during joint training.

Conversely, existing literature remains heavily siloed:
1. **Segmentation-focused topology papers** (e.g., persistent homology methods using Betti numbers \\(b_0, b_1\\)) enforce structural constraints but ignore downstream clinical diagnosis while suffering from high per-epoch computational costs.
2. **Multi-task networks** combine segmentation and classification on the ACDC dataset but lack explicit spatial boundary constraints, leaving them vulnerable to topological artifacts.

This project bridges these two domains by utilizing a shared-latent 2D network where enforcing spatial topological continuity natively regularizes the feature embeddings fed to the 5-class clinical disease classifier.

---

## 🎯 Literature-Derived Study Objectives

After reviewing recent literature (2021–2026) across medical image analysis and multi-task learning, this research is structured around four primary objectives:

1. **Multi-Task Architecture Design:** Construct a shared 2D encoder backbone with bifurcated output paths: a multi-class decoder for semantic segmentation (LV, RV, MYO) and a latent feature classifier for 5 clinical cardiomyopathy cohorts.
2. **Formulation of Topology-Preserving Loss L_{topo}:** Develop a lightweight, differentiable boundary penalty using 2D Sobel spatial gradient operators to penalize edge inconsistencies and disconnected regions without O(N^3) persistent homology loops.
3. **Quantifying Anatomical Viability:** Move beyond standard pixel-wise Dice metrics to evaluate structural continuity using Hausdorff Distance (**HD95**) and explicitly track topological violation rates.
4. **Proving the Joint-Learning Benefit:** Demonstrate mathematically and statistically that enforcing topologically sound segmentation masks acts as an implicit regularizer, reducing classifier cross-validation variance and improving clinical diagnostic accuracy.

---

## 🧮 Mathematical Formulation

The total joint optimization target is defined as:

\\[L_{total} = \alpha L_{seg} + \beta L_{class} + \gamma L_{topo}\\]

Where:
* **\\(L_{seg}\\)**: Standard Cross-Entropy / Soft Dice loss evaluated against 4-class ground truth masks.
* **\\(L_{class}\\)**: Categorical Cross-Entropy loss evaluated against 5 clinical disease classes.
* **\\(L_{topo}\\)**: The **Directional Gradient Boundary Penalty** calculated across anatomical channels \\(c \in \{\text{LV}, \text{RV}, \text{MYO}\}\\):

\\[L_{topo} = \sum_{c} \frac{1}{N} \sum_{i=1}^{N} \left\| \nabla P_i^c - \nabla Y_i^c \right\|^2\\]

Here, \\(\nabla P_i^c\\) and \\(\nabla Y_i^c\\) represent spatial gradient vectors extracted via fixed \\(3 \times 3\\) Sobel \\(X\\) and \\(Y\\) filters applied to predicted channel probabilities and one-hot ground truth masks.

---

## 📊 Target Performance Metrics

Derived from state-of-the-art benchmarks on the **ACDC (Automated Cardiac Diagnosis Challenge)** dataset:

### 1. Segmentation & Structural Continuity
| Anatomical Structure | Target Dice (End-Diastole) | Target Dice (End-Systole) | Target HD95 (mm) |
| :--- | :---: | :---: | :---: |
| **Left Ventricle (LV)** | \\(\ge 95.0\%\\) | \\(\ge 90.0\%\\) | \\(< 4.0\text{ mm}\\) |
| **Right Ventricle (RV)** | \\(\ge 92.0\%\\) | \\(\ge 86.0\%\\) | \\(< 6.5\text{ mm}\\) |
| **Myocardium (MYO)** | \\(\ge 89.0\%\\) | \\(\ge 89.0\%\\) | \\(< 4.5\text{ mm}\\) |

### 2. Clinical Classification & Reliability
* **5-Class Cardiomyopathy Accuracy:** Target \\(\ge 93.0\% - 95.0\%\\) across Normal, MINF, DCM, HCM, and ARV patient cohorts.
* **Topological Violation Rate:** Reduction from baseline \\(14\% - 18\%\\) down to **\\(< 0.5\%\\)** of total slices.
* **Classifier Cross-Validation Variance:** Variance reduction from \\(\pm 4.8\%\\) to **\\(\pm 1.2\%\\)**, proving joint stabilization.

---

## 📁 Repository Directory Structure

```text
cardiac-mri-topology/
├── .gitignore               # Excludes large .h5 files, checkpoints, caches
├── README.md                # Project documentation and theoretical background
├── requirements.txt         # Core dependencies (PyTorch, MONAI, h5py, etc.)
├── config.py                # Hyperparameters, paths, and device configuration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_phase1_demo.ipynb
└── src/
    ├── __init__.py
    ├── dataset.py           # ACDC HDF5 dataloader with dynamic interpolation & ID mapping
    ├── model.py             # SharedEncoderMultiTaskNet PyTorch architecture
    ├── loss.py              # DirectionalGradientLoss (L_topo) module
    ├── utils.py             # Plotting and metrics evaluation functions
    └── train.py             # Master multi-task joint optimization training loop
```

---

## 🗓️ 11-Month Execution & Publication Roadmap

* **Phase 1: Mathematical Novelty & (L_{topo} Formulation (Months 1–4 / Jun–Sep 2026)**
  * ACDC dataset preprocessing, HDF5 slice extraction, multi-task baseline setup, and Sobel gradient penalty integration.
  * *Target Publication 1:* **IEEE Transactions on Medical Imaging (TMI)** or **Medical Image Analysis**.
* **Phase 2: Multi-Task Framework & Joint Optimization (Months 5–8 / Oct 2026–Jan 2027)**
  * Integration of classification head, ablation study L_{topo} ON vs. OFF, and gradient norm tracking.
  * *Target Publication 2:* **IEEE Journal of Biomedical and Health Informatics (JBHI)** or **Machine Intelligence Research**.
* **Phase 3: Clinical Validation & Thesis Assembly (Months 9–11 / Feb–Apr 2027)**
  * Derived clinical marker calculation (EF, Myocardial Mass, Wall Thickness), final thesis defense.
  * *Target Publication 3:* **Computers in Biology and Medicine** or **Scientific Reports**.

---

## 🚀 Quickstart Guide

1. **Clone & Install Dependencies:**
   ```bash
   git clone https://github.com/your-username/cardiac-mri-topology.git
   cd cardiac-mri-topology
   pip install -r requirements.txt
   ```

2. **Execute Multi-Task Dry-Run:**
   ```bash
   python src/train.py
   ```

---