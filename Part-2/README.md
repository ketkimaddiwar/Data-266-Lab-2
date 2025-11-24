# GenAI Lab 2 — Part 2

This project fine-tunes **Stable Diffusion v1.5** using **LoRA (Low-Rank Adaptation)** on the **WikiArt** dataset to adapt the model for artistic style generation. It also includes evaluation metrics and visual comparisons between the baseline and fine-tuned models.

---

## Overview

The script automates the full pipeline of:
1. **Dataset preparation** (WikiArt via Hugging Face Datasets)
2. **Model setup** (Stable Diffusion 1.5 + LoRA using PEFT)
3. **Training loop** (fine-tuning only the LoRA parameters)
4. **Image generation** (baseline vs fine-tuned)
5. **Evaluation** using:
   - **Inception Score (IS)**
   - **CLIP similarity**
6. **Visualization** of generated images side by side.

---

##  Key Dependencies

| Library | Version (specified in script) |
|----------|------------------------------|
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| diffusers | 0.30.0 |
| transformers | 4.44.2 |
| accelerate | 1.0.1 |
| datasets | 3.0.1 |
| torchmetrics | 1.4.1 |
| pillow | 10.4.0 |
| safetensors | 0.4.5 |
| tqdm | 4.66.5 |
| scikit-image | 0.24.0 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| peft | 0.12.0 |
| matplotlib | latest |

> Install all dependencies using the command in the script:
> ```bash
> pip install -r requirements.txt
> ```
> or manually run the provided pip install lines inside the script.

---

## Script Structure

### 1. Configuration
The `Config` dataclass defines all hyperparameters:
- Model: `runwayml/stable-diffusion-v1-5`
- Dataset: `huggan/wikiart`
- Image size: `512x512`
- LoRA parameters: rank = 8, alpha = 16
- Training steps: 2000 (adjustable)
- Mixed precision: `fp16` (optional)

### 2. Dataset Preparation
- Loads the **WikiArt** dataset and builds captions dynamically from metadata (title, artist, style, genre).
- Normalizes and resizes images to 512×512.
- Creates `train` and `validation` splits and DataLoaders.

### 3. Model Setup (PEFT + LoRA)
- Loads Stable Diffusion 1.5 from Hugging Face.
- Freezes base UNet parameters.
- Injects LoRA layers into the attention projections using PEFT.

### 4. Training
- Optimizer: `AdamW`
- Scheduler: cosine with warmup
- Loss: MSE between predicted and target noise
- Uses gradient accumulation and AMP (fp16) for memory efficiency.
- Saves LoRA weights every `save_every` steps.

### 5. Inference
- Compares **baseline** and **LoRA fine-tuned** models on five prompts.
- Generates multiple seeds for each prompt.
- Saves outputs and logs in CSV format.

### 6. Evaluation
- Computes:
  - **Inception Score (IS)** — diversity and quality
  - **CLIP Similarity** — text–image alignment
- Saves quantitative metrics in `eval_metrics_summary.csv`.
- Plots qualitative comparison (Baseline vs Fine-tuned).

---

## Output Structure

```
sd15_wikiart_lora/
│
├── lora_ckpts/                 # Saved LoRA checkpoints
├── eval_generations/
│   ├── baseline_pretrained/    # Images from baseline model
│   └── finetuned_lora/         # Images from fine-tuned model
│   └── gen_log_all.csv         # Combined generation log
│
├── eval_metrics_summary.csv    # Quantitative evaluation
└── training logs
```

---

## Usage

1. **Run the script** in Colab or a local GPU environment:
   ```bash
   python genai_lab_2_part_2.py
   ```

2. **Adjust configurations** (e.g., dataset, steps, batch size) in the `Config` class.

3. **Check outputs**:
   - Generated images in `eval_generations/`
   - Metrics summary in `eval_metrics_summary.csv`

---

##  Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Inception Score (IS)** | Evaluates image realism and diversity. |
| **CLIP Similarity** | Measures semantic alignment between text prompts and generated images. |

---

## Example Prompts
- “Colorful abstract painting”
- “A cat wearing a crown”
- “Impressionist riverside garden with water lilies”
- “Surreal melting clocks in a desert landscape”
- “Baroque still life with fruits and dramatic lighting”

---

- Designed for educational use (GenAI Lab 2).
- Requires GPU (NVIDIA with ≥12GB VRAM recommended).
- Compatible with PyTorch AMP and CUDA acceleration.
- Avoid running on CPU due to heavy computational cost.
