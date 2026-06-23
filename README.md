# CardioDiffusionIA_Teven

**Research project focused on the application of deep learning and diffusion models for cardiovascular signal regeneration and ECG cardiac diagnostics.**

This repository contains the complete framework developed for the study, including source code, evaluation tools, scientific documentation, and experimental results.

---

## 📂 Repository Content
- `cardio_diffusion_v4.py`: Main implementation of the generative diffusion model for processing cardiac signals.
- `Evaluadorv4.py`: Standalone script to evaluate model performance, calculate metrics, and visualize results.
- `PAPER_FINAL_CARDIO_DIFFUSION.pdf`: Full scientific article detailing the theoretical foundations, methodology, and experimental findings.
- `requisitos.txt`: Complete list of Python dependencies required to run the project.

---

## ⚠️ Critical Performance Warning
**Please read before running the training loop.**

This model is computationally intensive and requires significant hardware resources.
- **Estimated training time:** Approximately **10+ hours** (depending on your specific GPU/CPU configuration).
- **Hardware used in this study:** *(Optional: Add your GPU here, e.g., NVIDIA RTX 3060)*.

If you do not have a dedicated GPU or sufficient time, it is strongly recommended to **skip the full training** and use `Evaluadorv4.py` directly to test the model's behavior on the provided results.

---

## 🚀 Getting Started
To replicate the environment and run the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TevenEu/CardioDiffusionIA_Teven.git
