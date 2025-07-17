# Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS

**Official implementation of the MICCAI 2025 (Early Accept, Top 9%) paper**

> **Status:** Code under active development

[Paper](https://arxiv.org/pdf/2503.01075) &nbsp;&nbsp;|&nbsp;&nbsp; [Contact Author](mailto:seunghoi.kim.17@ucl.ac.uk)

---

## Overview

Hallucinations—spurious structures not present in ground truth—pose a critical challenge in medical image reconstruction, particularly for data-driven conditional models. Our work investigates this phenomenon and introduces DynamicDPS, an innovative approach designed to mitigate hallucination while improving reconstruction fidelity and efficiency.

---

## Visual Comparisons

Below: Visual comparisons on real low-field MR scans. DynamicDPS demonstrates superior reconstruction quality with fewer hallucinated features.

![Visual comparisons on real low-field MR scans](images/visual_comparison.png)

---

## Method Overview

The schematic below illustrates our method (DynamicDPS) in comparison to traditional approaches. DynamicDPS achieves faster inference and avoids hallucination, outperforming standard conditional and diffusion models.

![Schematic overview: DynamicDPS vs. traditional approaches](images/method_overview.png)

---

## Usage

> **Note:** The final cleaned-up version of the code will be released soon.

### Training the Score-Matching Model
```bash
python image_train.py
```

### Solve Inverse Problems (e.g., Low-Field MRI Enhancement)
```bash
python test.py
```

---

## Citation

If you find this work useful, please consider citing:
```bibtex
@article{kim2025tackling,
  title={Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS},
  author={Kim, Seunghoi and Tregidgo, Henry FJ and Figini, Matteo and Jin, Chen and Joshi, Sarang and Alexander, Daniel C},
  journal={arXiv preprint arXiv:2503.01075},
  year={2025}
}
```
