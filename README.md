# Lightweight Tool Selector for Edge AI Agents

M.Tech Project 1 | Target: AAAI 2026 / EMNLP 2025

## Overview

This project compares ultra-lightweight BERT classifiers for tool selection in LLM agents on edge devices.

**Base Paper:** TinyAgent (EMNLP 2024) - https://arxiv.org/abs/2409.00608

## Models Compared

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| TinyBERT | 14M | huawei-noah/TinyBERT_General_4L_312D |
| MobileBERT | 25M | google/mobilebert-uncased |
| DeBERTa | 44M | microsoft/deberta-v3-small |
| DistilBERT | 66M | distilbert-base-uncased |

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n tool_selector python=3.10
conda activate tool_selector

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
cd src
python dataset.py  # Creates sample data in data/ folder
```

For full dataset, either:
- Download from TinyAgent GitHub (if available)
- Generate using GPT API (see implementation guide)

### 3. Train Models

```bash
# Train single model
python src/train.py --model tinybert --epochs 3

# Train all models
python src/train.py --model all --epochs 3
```

### 4. Benchmark

```bash
# Benchmark on CPU
python src/benchmark.py --model all --device cpu

# Benchmark on GPU (if available)
python src/benchmark.py --model all --device cuda
```

## Project Structure

```
project_code/
├── data/
│   ├── train.json      # Training data
│   ├── val.json        # Validation data
│   └── test.json       # Test data
├── models/
│   ├── tinybert/       # Saved TinyBERT model
│   ├── mobilebert/     # Saved MobileBERT model
│   ├── distilbert/     # Saved DistilBERT model
│   └── deberta/        # Saved DeBERTa model
├── src/
│   ├── dataset.py      # Dataset class
│   ├── model.py        # Model definitions
│   ├── train.py        # Training script
│   └── benchmark.py    # Benchmarking script
├── results/
│   └── benchmark_results.json
├── requirements.txt
└── README.md
```

## Dataset Format

```json
[
  {
    "query": "Send an email to John about the meeting",
    "tools": ["compose_new_email", "get_email_address"]
  },
  {
    "query": "Create a calendar event for tomorrow at 3pm",
    "tools": ["create_calendar_event"]
  }
]
```

## 16 Tools

1. compose_new_email
2. reply_to_email
3. forward_email
4. get_email_address
5. get_phone_number
6. send_sms
7. create_calendar_event
8. get_zoom_meeting_link
9. create_note
10. open_note
11. append_note_content
12. create_reminder
13. open_and_get_file_path
14. summarize_pdf
15. maps_open_location
16. maps_show_direction

## Expected Results

| Model | Latency (RPi) | Recall | Status |
|-------|---------------|--------|--------|
| TinyBERT | ~30ms | ~94% | ⚡ Fastest |
| MobileBERT | ~50ms | ~97% | ✅ Best Balance |
| DeBERTa | ~150ms | ~99% | 🐢 Baseline |
| DistilBERT | ~100ms | ~98% | 📊 High Accuracy |

## Citation

```bibtex
@inproceedings{your-paper-2025,
  title={Ultra-Lightweight Tool Selection for Edge-Deployed LLM Agents},
  author={Your Name},
  booktitle={AAAI/EMNLP 2025/2026},
  year={2025}
}
```

## References

- TinyAgent: https://github.com/SqueezeAILab/TinyAgent
- TinyBERT: https://arxiv.org/abs/1909.10351
- MobileBERT: https://arxiv.org/abs/2004.02984
- DistilBERT: https://arxiv.org/abs/1910.01108
- DeBERTa: https://arxiv.org/abs/2111.09543
