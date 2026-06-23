# CardioDiffusionIA_Teven

**Research project focused on the application of conditional Denoising Diffusion Probabilistic Models (DDPMs) for the generative reconstruction of missing electrocardiogram (ECG) segments and subsequent cardiac arrhythmia classification.**

This framework is designed as a signal-rescue strategy for ambulatory Holter monitoring. It utilizes a mask-guided 1D U-Net with attention mechanisms to recover clinically relevant ECG morphology (QRS complex, ST segment, and T wave) from artificially corrupted signals. The study not only evaluates reconstruction quality (using PRD, MSE, and Pearson correlation) but also validates the diagnostic utility of the regenerated signals through a downstream binary classifier, achieving a **95.1% arrhythmia sensitivity** on the MIT-BIH Arrhythmia Database.

This repository contains the complete framework developed for the study, including source code, evaluation tools, and the full scientific documentation.

---

## Repository Content
- `cardio_diffusion_v4.py`: Main implementation of the conditional DDPM with a 1D U-Net backbone for two-lead ECG reconstruction.
- `Evaluadorv4.py`: Standalone script to evaluate model performance (PRD, MSE, MAE, RMSE, Pearson r), calculate classification metrics, and visualize results.
- `PAPER_FINAL_CARDIO_DIFFUSION.pdf`: Full scientific article detailing the theoretical foundations, methodology (hybrid loss functions, anatomical masking), and experimental findings.
- `requisitos.txt`: Complete list of Python dependencies required to run the project.

---

## Critical Performance Warning
**Please read before running the training loop.**

This model is computationally intensive and requires significant hardware resources.
- **Estimated training time:** Approximately **10+ hours** (depending on your specific GPU/CPU configuration).
- **Hardware used in this study:** *(Optional: Add your GPU here, e.g., NVIDIA RTX 4050 6GB)*.

If you do not have a dedicated GPU or sufficient time, it is strongly recommended to **skip the full training** and use `Evaluadorv4.py` directly to test the model's behavior on the provided results.

---

## Getting Started
To replicate the environment and run the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TevenEu/CardioDiffusionIA_Teven.git
