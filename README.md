
# MemeShield: A Framework for Toxic Meme Moderation

## Overview
MemeShield is a vision-language framework designed to address toxic content in memes by combining advanced machine learning models. The system integrates Large Language Models (LLMs) and Vision Transformers to:
- Identify **targets** (e.g., gender, race, religion).
- Classify **attack types** (e.g., mocking, dehumanizing).
- Generate tailored **interventions** to mitigate harm.

The project utilizes the **MemeProtect dataset**, a collection of 3,244 annotated memes in a code-switched Hindi-English context, enabling a holistic approach to multimodal content moderation.
---

## Features
- **Multimodal Analysis**: Leverages both textual and visual elements of memes using models like LLaMA, LLaMA2, FLAN-T5, Dolly, RedPajama, ViT, and SwinTransformer.
- **Efficient Fine-Tuning**: Implements QLoRA adapters to reduce computational requirements while maintaining high model accuracy.
- **Comprehensive Evaluation**: Validates outputs using ROUGE, BLEU, and human evaluation metrics (fluency, adequacy, persuasiveness).
---
# Project Structure

```
Github_intervention/
├── datasets/                      # Contains dataset files for training, testing, and validation
│   ├── test.csv                   # Testing dataset
│   ├── target_attack_type_intervention.csv # Data for target, attack type, and intervention mapping
│   ├── attack_intervention.csv    # Data mapping attack types to interventions
│   ├── target_intervention.csv    # Data mapping targets to interventions
│   └── Entire_Data.csv            # Consolidated dataset for the project
├── results/                       # Generated results and processed datasets
│   ├── generated_data_llama - generated_data_llama.csv.csv # Generated results from LLaMA
│   ├── Flan_Test_TAI.csv          # Test results from FLAN-T5
│   ├── llama2_test_TI.csv         # Test results for LLaMA-2 (target, intervention)
│   ├── Flan_Test_I.csv            # FLAN-T5 intervention results
│   ├── llama2_test_TIA.csv        # LLaMA-2 test results (target, intervention, attack)
│   ├── generated_data_llama.csv   # Additional results from LLaMA
│   ├── generated_data_flan.csv    # Additional results from FLAN-T5
│   ├── Flan_Test_AI.csv           # Test results from FLAN-T5 (attack, intervention)
│   └── Flan_Test_TI.csv           # Test results from FLAN-T5 (target, intervention)
├── scripts/                       # Python scripts for model training and evaluation
│   ├── llama2_multimodal_i.py     # LLaMA-2 script for intervention-only tasks
│   ├── llama2_multimodal_ai.py    # LLaMA-2 script for attack-intervention tasks
│   ├── llama2_multimodal_ti.py    # LLaMA-2 script for target-intervention tasks
│   ├── llama2_multimodal_tai.py   # LLaMA-2 script for target-attack-intervention tasks
│   ├── dolly_adapter_test_tai.py  # Dolly model script for full multimodal tasks
│   ├── dolly_adapter_test_i.py    # Dolly model script for intervention-only tasks
│   ├── dolly_adapter_test.py      # Generic Dolly test script
│   ├── dolly_adapter.py           # Dolly model adapter implementation
│   ├── dolly_adapter_test_ai.py   # Dolly model script for attack-intervention tasks
│   └── dolly_adapter_test_ti.py   # Dolly model script for target-intervention tasks
├── requirements.txt               # Dependencies for the project
├── Readme.md                      # Project documentation
└── LLM Agents.pdf                 # Supplementary documentation for LLM agents
```
---

## Setup
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training or inference scripts(needs CUDA setup):
   ```bash
   python scripts/llama2_multimodal_tai.py
   ```

---

## Usage
1. **Prepare Dataset**: Place the dataset files in the `datasets/` directory.
2. **Model Training**: Use scripts in the `scripts/` folder to train models on target, attack, and intervention tasks.
3. **Evaluation**: Generate and evaluate outputs using provided metrics.

---

## Results
- Multimodal models (LLMs + Vision Transformers) outperform unimodal baselines.
- High human evaluation scores for intervention quality: fluency (4.91), adequacy (4.87), persuasiveness (4.46).
---

## Contributors
- Developed by Sri Vaibhav Devarasetty from CSE Department, IIT Patna.
- Supervised by Dr. Sriparna Saha, IIT Patna.
