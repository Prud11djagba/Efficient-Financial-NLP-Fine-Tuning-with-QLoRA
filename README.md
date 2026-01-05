# Efficient Quantized Low-Rank Adaptation of Large Language Models for Financial NLP on the FLARE Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)

> **FinQLoRA**: Multi-Adapter QLoRA Fine-tuning of Meta-Llama-3.1-8B-Instruct for 5 Financial NLP Tasks on 8 FLARE Benchmark Datasets

## 📋 Overview

This repository implements the methodology described in our paper: **"Efficient Quantized Low-Rank Adaptation of Large Language Models for Financial NLP on the FLARE Benchmark"**.

We present **FinQLoRA**, a parameter-efficient fine-tuning approach using QLoRA (4-bit quantization + LoRA) on Meta-Llama-3.1-8B-Instruct, achieving competitive performance with **only 0.17-0.69% trainable parameters** on 5 financial tasks from the FLARE benchmark.

### Key Achievements

- 🎯 **Matches FinMA-30B** on ConvFinQA (40.1% EM) with **73% fewer parameters**
- 💰 **$50 total training cost** (~50,000× cheaper than BloombergGPT's $2.6M)
- ⚡ **9 hours training time** for all 5 tasks on single A100 40GB GPU
- 📊 **8 datasets** across 5 financial NLP tasks from FLARE benchmark

### Tasks & Datasets (5 Tasks, 8 Datasets)

| Task | Abbr | # Datasets | Datasets | Max Length | LoRA Config |
|------|------|-----------|----------|------------|-------------|
| Sentiment Analysis | SA | 2 | FPB, FiQA-SA | 512 | r=8, α=16 |
| Headline Classification | HC | 1 | Headlines | 512 | r=8, α=16 |
| Named Entity Recognition | NER | 1 | FLARE-NER | 1024 | r=16, α=32 |
| Question Answering | QA | 2 | FinQA, ConvFinQA | 2048 | r=16, α=32 |
| Stock Movement Prediction | SMP | 2 | Stock-CIKM, Stock-BigData | 1800 | r=32, α=64 |

## 📁 Project Structure

```
Efficient-Financial-NLP-Fine-Tuning-with-QLoRA/
├── configs/
│   ├── model_config.yaml              # Base model & quantization settings
│   └── tasks/                         # Task-specific configurations
│       ├── sa_config.yaml             # Sentiment Analysis
│       ├── hc_config.yaml             # Headline Classification
│       ├── ner_config.yaml            # Named Entity Recognition
│       ├── qa_config.yaml             # Question Answering
│       └── smp_config.yaml            # Stock Movement Prediction
│
├── data/
│   ├── formatted/                     # Llama 3.1 formatted datasets
│   │   ├── sa/merged/                 # FPB + FiQA-SA
│   │   ├── hc/merged/                 # Headlines
│   │   ├── ner/merged/                # FLARE-NER
│   │   ├── qa/merged/                 # FinQA + ConvFinQA
│   │   └── smp/merged/                # Stock-CIKM + Stock-BigData
│   ├── dataset_config.json            # Dataset mappings & specifications
│   ├── llama_template.txt             # Llama 3.1 chat template
│   └── metadata.json                  # Dataset statistics (auto-generated)
│
├── src/
│   ├── data/
│   │   └── dataset_loader.py          # Dataset loading & validation
│   ├── models/
│   │   └── qlora_model.py             # QLoRA with 4-bit quantization
│   ├── training/
│   │   ├── trainer.py                 # TaskTrainer for single-task training
│   │   └── callbacks.py               # Monitoring callbacks
│   ├── evaluation/
│   │   └── evaluator.py               # SOTA-comparable evaluation metrics
│   └── utils/
│       └── training_monitor.py        # Real-time metrics tracking
│
├── scripts/
│   ├── train.py                       # Train single task
│   ├── train_all.py                   # Batch training (all tasks)
│   ├── eval_model.py                  # Evaluate single task
│   ├── eval_all_models.py             # Batch evaluation
│   └── verify_datasets.py             # Dataset format validation
│
├── outputs/
│   ├── adapters/                      # Trained LoRA adapters (~27MB each)
│   │   ├── sa_adapter/
│   │   ├── hc_adapter/
│   │   ├── ner_adapter/
│   │   ├── qa_adapter/
│   │   └── smp_adapter/
│   ├── evaluations/                   # Evaluation results (JSON)
│   └── logs/                          # Training logs & metrics
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Installation

**Requirements:**
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- **40GB+ VRAM recommended** (A100 40GB tested)
- Minimum 16GB RAM

```bash
# Clone repository
git clone https://github.com/AbdelkaderYS/Efficient-Financial-NLP-Fine-Tuning-with-QLoRA.git
cd Efficient-Financial-NLP-Fine-Tuning-with-QLoRA

# Install dependencies
pip install -r requirements.txt
```

**Core Dependencies:**
```
torch>=2.1.0
transformers>=4.36.2
datasets>=2.16.1
peft>=0.7.1
trl>=0.8.1
bitsandbytes>=0.41.3
scikit-learn>=1.3.0
```

### 2. Dataset Preparation

The datasets are automatically downloaded from Hugging Face Hub during the formatting pipeline.

**Run the formatting pipeline** (from Colab/Jupyter notebook):

```python
# Execute Cell 1: Project Setup
# Execute Cell 2: Dataset Formatting

# This will:
# 1. Download 8 datasets from FLARE benchmark
# 2. Format into Llama 3.1 instruction format
# 3. Apply text normalization
# 4. Save to data/formatted/{task}/merged/
```

**Or run the full notebook:**
```bash
jupyter notebook Efficient-Financial-NLP-Fine-Tuning-with-QLoRA.ipynb
```

### 3. Verify Datasets

**Before training, validate dataset format:**

```bash
# Verify all tasks
python scripts/verify_datasets.py

# Verify specific task
python scripts/verify_datasets.py --task sa
```

### 4. Training

#### Single Task Training

```bash
# Train Sentiment Analysis
python scripts/train.py --task sa

# Train Question Answering
python scripts/train.py --task qa

# Train Named Entity Recognition
python scripts/train.py --task ner
```

#### Batch Training (All Tasks)

```bash
python scripts/train_all.py
```

**Training Output:**
- Adapters saved to: `outputs/adapters/{task}_adapter/final_adapter/`
- Logs saved to: `outputs/adapters/{task}_adapter/metrics.json`
- Plots saved to: `outputs/adapters/{task}_adapter/training_metrics.png`

### 5. Evaluation

#### Single Task Evaluation

```bash
# Evaluate Sentiment Analysis
python scripts/eval_model.py --task sa

# Evaluate with limited samples (faster)
python scripts/eval_model.py --task ner --samples 100

# Disable BERTScore (faster for QA)
python scripts/eval_model.py --task qa --no-bertscore
```

#### Batch Evaluation (All Tasks)

```bash
python scripts/eval_all_models.py
```

**Evaluation Output:**
- Results saved to: `outputs/evaluations/{task}/{dataset}_results.json`
- Summary saved to: `outputs/evaluations/{task}/summary.json`

## ⚙️ Task-Specific Configurations

### Configuration Files

Each task has an independent configuration in `configs/tasks/{task}_config.yaml`:

#### Sentiment Analysis (SA)

```yaml
task_name: sentiment_analysis
adapter_name: sa_adapter
dataset_path: data/formatted/sa/merged
max_sequence_length: 512

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05

training_args:
  num_epochs: 3
  learning_rate: 0.0002
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 8
```

#### Question Answering (QA)

```yaml
task_name: question_answering
adapter_name: qa_adapter
dataset_path: data/formatted/qa/merged
max_sequence_length: 2048

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training_args:
  num_epochs: 2
  learning_rate: 0.0002
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
```

#### Stock Movement Prediction (SMP)

```yaml
task_name: stock_movement_prediction
adapter_name: smp_adapter
dataset_path: data/formatted/smp/merged
max_sequence_length: 1800

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.10

training_args:
  num_epochs: 2
  learning_rate: 0.0002
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
```

### Complete Adapter Configurations

| Task | Rank (r) | Alpha (α) | Dropout | Target Modules | Trainable Params |
|------|----------|-----------|---------|----------------|------------------|
| SA   | 8        | 16        | 0.05    | q,v,k,o proj   | 6.4M (0.08%)     |
| HC   | 8        | 16        | 0.05    | q,v,k,o proj   | 6.4M (0.08%)     |
| NER  | 16       | 32        | 0.10    | all linear     | 13.6M (0.17%)    |
| QA   | 16       | 32        | 0.05    | q,v,k,o proj   | 13.6M (0.17%)    |
| SMP  | 32       | 64        | 0.10    | all linear     | 55.6M (0.69%)    |

## 📊 Performance Results

### Main Results

Performance comparison with state-of-the-art models on FLARE benchmark:

| Task | Dataset | Metric | FinQLoRA | Best Baseline | Difference |
|------|---------|--------|----------|---------------|------------|
| **SA** | FPB | F1 | **84.3%** | 88.0% (FinMA-30B) | -3.7 |
| **SA** | FiQA-SA | F1 | **83.0%** | 88.0% (FinMA-30B) | -5.0 |
| **HC** | Headlines | Acc | **92.8%** | 98.0% (FinMA-7B) | -5.2 |
| **NER** | FLARE-NER | F1 | **58.1%** | 75.0% (FinMA-7B) | -16.9 |
| **QA** | FinQA | EM | **12.0%** | 11.0% (FinMA-30B) | **+1.0** |
| **QA** | ConvFinQA | EM | **40.1%** | 40.0% (FinMA-30B) | **+0.1** |
| **SMP** | Stock-CIKM | Acc | **56.08%** | 56.0% (FinMA-7B) | **+0.08** |
| **SMP** | Stock-BigData | Acc | **54.96%** | 49.0% (FinMA-30B) | **+5.96** |

*Baseline results from PIXIU benchmark [Xie et al., 2023]*

### Detailed Performance Breakdown

#### Classification Tasks

**Sentiment Analysis:**
- FPB: F1=84.3%, MCC=0.724, Precision=84.1%, Recall=84.5%
- FiQA-SA: F1=83.0%, MCC=0.687, Precision=82.8%, Recall=83.2%

**Headline Classification:**
- Accuracy=92.8%, MCC=0.831, Precision=94.6%, Recall=82.0%

#### Named Entity Recognition

| Entity Type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|-----|---------|
| PER         | 79.5%     | 73.0%  | **76.1%** | 122 |
| ORG         | 26.0%     | 30.2%  | 28.0% | 43  |
| LOC         | 41.2%     | 22.6%  | 29.2% | 31  |
| **Overall** | **60.9%** | **55.6%** | **58.1%** | **196** |

**Note: Performance disparity reflects strong class imbalance and small dataset size*

#### Question Answering

**FinQA (Single-turn):**
- Official EM: 12.0%
- Numerical EM (±0.01): 43.6%
- BERTScore F1: 83.9%

**ConvFinQA (Multi-turn):**
- Official EM: **40.1%**
- Numerical EM (±0.01): 49.7%
- BERTScore F1: 86.3%

**Key Finding:** 3.3× improvement from single-turn to multi-turn QA demonstrates effective use of conversational context.

#### Stock Movement Prediction

| Dataset | Accuracy | Precision | Recall | MCC |
|---------|----------|-----------|--------|-----|
| Stock-CIKM | 56.08% | 58.3% | 85.4% | 0.012 |
| Stock-BigData | 54.96% | 55.6% | 92.3% | 0.022 |

*Note: Low MCC scores indicate class imbalance bias (consistent with SOTA models)*

### Comparison with State-of-the-Art (Table VII from Paper)

| Model | Params | Method | FPB (F1) | Headlines (Acc) | NER (F1) | FinQA (EM) | ConvFinQA (EM) | CIKM18 (Acc) |
|-------|--------|--------|----------|-----------------|----------|------------|----------------|--------------|
| **FinQLoRA** | **8B** | **PEFT** | **84.3** | **92.8** | **58.1** | **12.0** | **40.1** | **56.08** |
| FinMA-7B | 7B | Full FT | 86.0 | 98.0 | 75.0 | 6.0 | 25.0 | 56.0 |
| FinMA-30B | 30B | Full FT | 88.0 | 97.0 | 62.0 | 11.0 | **40.0** | 43.0 |
| BloombergGPT | 50B | Pretrain | 51.0 | 82.0 | 61.0 | — | 43.0 | — |
| ChatGPT | — | Zero-shot | 75.0 | 77.0 | 77.0 | 58.0 | 60.0 | 55.0 |
| GPT-4 | — | Zero-shot | 86.0 | 86.0 | 83.0 | 63.0 | 76.0 | 57.0 |

**Key Observations:**
1. ✅ **Matches FinMA-30B** on ConvFinQA (40.1% vs 40.0%) with **73% fewer parameters**
2. ✅ **Surpasses FinMA-30B** on Stock-BigData (+5.96 points)
3. ✅ **Competitive** on classification tasks (within 4-5% of specialized models)

## ⚡ Computational Efficiency

### Training Efficiency (Table VI from Paper)

| Task | Time (min) | Steps | GPU Base (GB) | GPU Peak (GB) | Memory Overhead | Trainable Params | Loss Reduction |
|------|-----------|-------|---------------|---------------|-----------------|------------------|----------------|
| SA   | 25        | 605   | 5.37          | 8.27          | 54%             | 6.4M (0.08%)     | 70.6%          |
| HC   | 104       | 2,247 | 5.37          | 8.27          | 54%             | 6.4M (0.08%)     | 88.7%          |
| NER  | 8         | 130   | 5.43          | 13.75         | 153%            | 13.6M (0.17%)    | 87.5%          |
| QA   | 295       | 1,894 | 5.41          | **37.10**     | **586%**        | 13.6M (0.17%)    | 58.0%          |
| SMP  | 115       | 1,038 | 5.80          | 36.97         | 537%            | 55.6M (0.69%)    | 41.5%          |
| **Avg** | **109** | **1,183** | **5.48** | **20.87** | **277%** | **19.1M (0.24%)** | **69.3%** |

**Key Metrics:**
- ✅ **Total training time:** 9 hours (547 minutes) for all 5 tasks
- ✅ **Total cost:** ~$50 (Google Colab Pro / Cloud GPU)
- ✅ **Single GPU:** All tasks fit on A100 40GB
- ✅ **Trainable parameters:** 0.08% - 0.69% (average 0.24%)

### Memory Overhead Explanation

Memory overhead varies by sequence length:
- **Short sequences (SA, HC):** 54% overhead
- **Medium sequences (NER):** 153% overhead
- **Long sequences (QA, SMP):** 537-586% overhead

*Overhead driven by gradient accumulation during backpropagation, not model size*

## 💻 Usage Examples

### Python API

```python
from pathlib import Path
from src.training.trainer import TaskTrainer
from src.models.qlora_model import QLoRAModel
from src.evaluation.evaluator import SOTAComparableEvaluator

# 1. Train a task
trainer = TaskTrainer(task_key="sa")
model, tokenizer = trainer.train()

# 2. Evaluate trained adapter
evaluator = SOTAComparableEvaluator(
    task_key="sa",
    adapter_path="outputs/adapters/sa_adapter/final_adapter",
    batch_size=8
)
results = evaluator.evaluate_all_datasets()

print(f"Accuracy: {results['fpb']['accuracy']:.4f}")
print(f"F1-score: {results['fpb']['f1']:.4f}")
```

### Command Line

```bash
# Train with default config
python scripts/train.py --task qa

# Evaluate with custom settings
python scripts/eval_model.py \
    --task qa \
    --adapter outputs/adapters/qa_adapter/final_adapter \
    --batch-size 4 \
    --samples 500 \
    --no-bertscore
```

### Load Trained Adapter for Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(
    model,
    "outputs/adapters/sa_adapter/final_adapter"
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Inference
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

What is the sentiment of this sentence? Answer with: positive, negative, or neutral.<|eot_id|><|start_header_id|>user<|end_header_id|>

The company reported record profits this quarter.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: positive
```

## 🔧 Advanced Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true

lora_common:
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
```

### Training Configuration

```yaml
training_args:
  optim: paged_adamw_8bit
  lr_scheduler_type: cosine_with_restarts
  warmup_ratio: 0.03
  weight_decay: 0.01
  fp16: false
  bf16: true
  gradient_checkpointing: true
  max_grad_norm: 1.0
```

## 📝 Dataset Format

All datasets are formatted into **Llama 3.1 chat template**:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}<|eot_id|>
```

**Example (Sentiment Analysis):**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

What is the sentiment of this sentence? Answer with: positive, negative, or neutral.<|eot_id|><|start_header_id|>user<|end_header_id|>

The company reported strong earnings growth.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

positive<|eot_id|>
```

### Tokenization Statistics (Table I from Paper)

| Task | Dataset | Train | Test | Mean Tokens | P95 Tokens | Max Length | Truncation (%) |
|------|---------|-------|------|-------------|------------|------------|----------------|
| SA | FPB | 3,099 | 970 | 218 | 246 | 512 | 0.0 |
| SA | FiQA-SA | 750 | 235 | 217 | 246 | 512 | 0.0 |
| HC | Headlines | 71,883 | 20,547 | 63 | 73 | 512 | 0.0 |
| NER | FLARE-NER | 398 | 98 | 149 | 264 | 1024 | 0.0 |
| QA | FinQA | 6,251 | 1,147 | 938 | 1,376 | 2,048 | 0.6 |
| QA | ConvFinQA | 8,891 | 1,490 | 974 | 1,457 | 2,048 | 0.6 |
| SMP | Stock-CIKM | 3,396 | 1,143 | 1,579 | 1,752 | 1,800 | 2.7 |
| SMP | Stock-BigData | 4,897 | 1,472 | 1,513 | 1,685 | 1,800 | 0.6 |

## 🐛 Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors during training:

1. **Reduce batch size:**
```yaml
per_device_train_batch_size: 2  # Reduce from 8
gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

2. **Reduce max sequence length** (if truncation < 10%):
```yaml
max_sequence_length: 1536  # For QA, reduce from 2048
```

3. **Enable gradient checkpointing** (already enabled by default)

### Slow Evaluation

If evaluation is too slow:

```bash
# Disable BERTScore computation
python scripts/eval_model.py --task qa --no-bertscore

# Evaluate on subset
python scripts/eval_model.py --task sa --samples 200
```

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Ensure you're in project root
cd /path/to/Efficient-Financial-NLP-Fine-Tuning-with-QLoRA

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Dataset Not Found

If datasets are missing:

```bash
# Re-run formatting pipeline (Cell 2 in notebook)
# Or manually verify:
python scripts/verify_datasets.py
```

## 📚 Citation

If you use this code or our results in your research, please cite our paper:

```bibtex
@article{djagba2025finqlora,
  title={Efficient Quantized Low-Rank Adaptation of Large Language Models for Financial NLP on the FLARE Benchmark},
  author={Djagba, Prudence and Younoussi Saley, Abdel Kader and Zeleke, Aklilu},
  journal={Submitted to Conference},
  year={2025}
}
```

### Referenced Works

```bibtex
@inproceedings{xie2023pixiu,
  title={PIXIU: A large language model, instruction data and evaluation benchmark for finance},
  author={Xie, Qianqian and others},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2023}
}

@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient finetuning of quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={NeurIPS},
  year={2023}
}

@article{wu2023bloomberggpt,
  title={BloombergGPT: A large language model for finance},
  author={Wu, Shijie and others},
  journal={arXiv preprint arXiv:2303.17564},
  year={2023}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📧 Contact

- **Prudence Djagba** - djagbapr@msu.edu (Michigan State University)
- **Abdel Kader Younoussi Saley** - saley.younoussi@aims.ac.rw (AIMS Rwanda)
- **Aklilu Zeleke** - zeleke@msu.edu (Michigan State University)


## 🙏 Acknowledgments

- FLARE benchmark by [Xie et al. (2023)](https://github.com/chancefocus/PIXIU)
- QLoRA implementation by [Dettmers et al. (2023)](https://github.com/artidoro/qlora)
- Hugging Face for transformers and PEFT libraries
- Meta for Llama 3.1 model

---

**Paper Status:** Submitted to Conference (2025)
**Code:** [GitHub Repository](https://github.com/AbdelkaderYS/Efficient-Financial-NLP-Fine-Tuning-with-QLoRA)
**Last Updated:** December 2024
