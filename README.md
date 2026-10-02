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
