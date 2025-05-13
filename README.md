# 🌀 Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS

Official implementation of the MICCAI 2025 (Early Accept, Top 9%) paper:  
**"Mitigating Hallucination from Conditional Models for MRI Reconstruction with DynamicDPS"**

[📄 Paper](https://arxiv.org/pdf/2503.01075) &nbsp;&nbsp;|&nbsp;&nbsp; [📧 Contact Author](mailto:seunghoi.kim.17@ucl.ac.uk)

---

## 🚀 Overview

Hallucinations are spurious structures not present in the ground truth, posing a critical challenge in medical image reconstruction, especially for data-driven conditional models. We hypothesize that combining an unconditional diffusion model with data consistency, trained on a diverse dataset, can reduce these hallucinations. Based on this, we propose DynamicDPS, a diffusion-based framework that integrates conditional and unconditional diffusion models to enhance low-quality medical images while systematically reducing hallucinations. Our approach first generates an initial reconstruction using a conditional model, then refines it with an adaptive diffusion-based inverse problem solver. DynamicDPS skips early stage in the reverse process by selecting an optimal starting time point per sample and applies Wolfe's line search for adaptive step sizes, improving both efficiency and image fidelity. Using diffusion priors and data consistency, our method effectively reduces hallucinations from any conditional model output. We validate its effectiveness in Image Quality Transfer for low-field MRI enhancement. Extensive evaluations on synthetic and real MR scans, including a downstream task for tissue volume estimation, show that DynamicDPS reduces hallucinations, improving relative volume estimation by over 15% for critical tissues while using only 5% of the sampling steps required by baseline diffusion models. As a model-agnostic and fine-tuning-free approach, DynamicDPS offers a robust solution for hallucination reduction in medical imaging. The code will be made publicly available upon publication.

---

## 🛠️ Usage

> 🔧 Note: The final cleaned-up version of the code will be released soon.

### 🧑‍🏫 Training the Score-Matching Model
```bash
python image_train.py

### 🖼️ Sampling Random Images from the Diffusion Model
python image_sample_all.py

### 🔄 Solve Inverse Problems (e.g., Low-Field MRI Enhancement)
python image_sample.py

### 📌 Citation
If you find this work useful, please consider citing us (BibTeX coming soon).


