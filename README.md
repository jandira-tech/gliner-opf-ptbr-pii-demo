---
title: OPF PT-BR PII Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.40.0
app_file: app.py
pinned: false
short_description: PII em PT-BR — openai/privacy-filter fine-tune
hf_oauth: false
license: apache-2.0
models:
- arthrod/gliner-opf-ptbr-pii-v1
datasets:
- arthrod/gliner-opf-ptbr-pii-bench-v1
---

# OPF PT-BR PII Detector

Finds personal and LGPD-sensitive data in Brazilian Portuguese text. It runs a fine-tune of OpenAI
Privacy Filter (not a GLiNER model, despite the repo name), trained and evaluated on synthetic data;
the benchmark and its limitations are in [the dataset card](https://huggingface.co/datasets/arthrod/gliner-opf-ptbr-pii-bench-v1).

Fine-tune do [openai/privacy-filter](https://github.com/openai/privacy-filter) treinado em **914 mil amostras** de PII em português brasileiro (texto natural autêntico das fontes upstream). 73 labels (24 PT-BR canônicos + 47 cross-source multilíngua).

## Recipe

- Backbone: openai/privacy-filter (8-layer sparse-MoE transformer, 128 experts, top-4 routing)
- Schedule: 3 épocas × 3 saves por época (9 checkpoints sequenciais)
- Optimizer: AdamW lr 1e-5, wd 0.01, max-grad-norm 1.0
- Batch: 32 windows × grad-accum 4 (efetivo 128)
- Context: n-ctx 256, bf16
- Hardware: AMD MI300X / ROCm 7.2

## Performance (vs GLiNER series, mesma val 5k PT-BR + 24 labels)

| Modelo | partial F1 | exact F1 |
|---|---|---|
| **opf-finetune (este)** | **0.897** | 0.853 |
| mmBERT-small × 3 | 0.888 | **0.870** |
| ettin-68m | 0.822 | 0.789 |
| ettin-32m | 0.808 | 0.769 |

Dados completos no [model card](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1).

## Related repos

| Repo | Type | Role |
|---|---|---|
| [arthrod/gliner-ettin-32m-ptbr-pii-full-3x-v1](https://huggingface.co/arthrod/gliner-ettin-32m-ptbr-pii-full-3x-v1) | model | GLiNER on `jhu-clsp/ettin-encoder-32m`, full ~984k-row train set, 69,000 steps. Smallest model in the family. |
| [arthrod/gliner-ettin-68m-ptbr-pii-full-3x-v1](https://huggingface.co/arthrod/gliner-ettin-68m-ptbr-pii-full-3x-v1) | model | GLiNER on `jhu-clsp/ettin-encoder-68m`, full ~984k-row train set, 69,000 steps (trainer state uploaded through step 58,650). |
| [arthrod/gliner-ettin-68m-ptbr-pii-top50k-v1](https://huggingface.co/arthrod/gliner-ettin-68m-ptbr-pii-top50k-v1) | model | GLiNER on `jhu-clsp/ettin-encoder-68m`, 50k curated subset, 11,700 steps. Ablation: shows the curated subset under-generalizes. |
| [arthrod/gliner-mmbert-small-ptbr-pii-full-3x-v1](https://huggingface.co/arthrod/gliner-mmbert-small-ptbr-pii-full-3x-v1) | model | GLiNER on `jhu-clsp/mmBERT-small`, full train set + 2k Ritápolis rows, 69,000 steps. Strongest GLiNER model of the family. |
| [arthrod/gliner-opf-ptbr-pii-v1](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1) | model | Fine-tune of OpenAI Privacy Filter (not GLiNER despite the name). Best partial F1 on the in-distribution val set and by far the fewest false positives on no-PII text. |
| [arthrod/gliner-opf-ptbr-pii-bench-v1](https://huggingface.co/datasets/arthrod/gliner-opf-ptbr-pii-bench-v1) | dataset | Benchmark: predictions and metrics for the mmBERT-small and Privacy Filter models (plus two unreleased ettin variants) on the PT-BR val set, held-out sources and no-PII spam/phishing text. |
| **arthrod/gliner-opf-ptbr-pii-demo** (this repo) | space | Gradio demo running the Privacy Filter fine-tune. |
| [arthrod/gliner_eval_folder](https://huggingface.co/datasets/arthrod/gliner_eval_folder) | dataset | Earlier (Feb 2026) sweep: 18 GLiNER models x 9 eval sets. |
| [arthrod/gliner_review_comparison](https://huggingface.co/datasets/arthrod/gliner_review_comparison) | dataset | Earlier GLiNER PII review/comparison data. |
