---
title: OPF PT-BR PII Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.13.0
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

> **Redesenho (este repositório).** O front-end Gradio foi substituído pelo design
> do design-system Arthur Souza Rodrigues (`OPF PT-BR PII - Demo.html`), agora
> servido por uma API FastAPI que chama o runtime OPF real. Nada é simulado:
> todo span exibido vem de `arthrod/gliner-opf-ptbr-pii-v1`.
>
> - `app.py` — servidor (FastAPI + host Gradio, compatível com ZeroGPU).
> - `opf_service.py` — taxonomia, categorias, placeholders e pós-processamento
>   dos spans (sem dependência de ML, testável sem GPU).
> - `opf_examples.py` — catálogo de exemplos (6 do design + 21 do app original).
> - `static/` — frontend do design (`index.html`, CSS, fontes, `demos/opf-app.js`).
> - `tests/` — suíte pytest (18 testes) com runtime falso.

## Arquitetura e deployment

O Space usa Gradio Server para servir a interface e enfileirar inferências em ZeroGPU. O modelo é carregado no processo do Space e a inferência usa `spaces.GPU`.

| Rota | Função |
|---|---|
| `/` e `/assets/*` | Interface, estilos e fontes |
| `/api/meta` | Rótulos, categorias e exemplos |
| `/gradio_api/call/analyze` | Análise enfileirada e resultado via eventos |
| `/api/health` | Estado de carregamento do modelo |

Cada análise recebe o texto, os rótulos e a opção de sobreposição. O aplicativo não mantém um cache de textos enviados. Como na interface original, uma seleção vazia desativa o filtro e considera todos os rótulos.

### O que mudou em relação ao design original

1. **Sem anotações fictícias.** O `demos/pii-data.js` (mock offline com spans
   escritos à mão) foi removido; os spans vêm do modelo.
2. **Contagem de rótulos correta.** O checkpoint tem 71 rótulos PII; as "73
   classes" do label space incluem `O` (background) e o easter egg
   `berco_de_tiradentes`. A UI mostra "71" e deriva o número do catálogo.
3. **Taxonomia completa.** As 12 categorias do design cobriam 59 rótulos; agora
   as 9 categorias cobrem os 71 (o resto do app original foi preservado).
4. **Overlap funcional.** O toggle "descartar spans sobrepostos do mesmo
   rótulo" passou a ter efeito real (como em `opf`), em vez de ser decorativo.
5. **Placeholders numerados** (`[CPF_1]`, `[NOME_2]`) mantidos do design; o
   placeholder bruto do modelo (`<CPF_DOCUMENT_NUMBER>`) vem no payload.
6. **Exemplos preservados.** Os 6 exemplos do design (ids `whats`, `peticao`,
   `prontuario`, `cv`, `ticket`, `daiana` — deep links `?ex=` intactos) e os 21
   do app original, com o filtro de rótulos por exemplo.
7. **Acessibilidade e responsividade.** `aria-pressed`/`aria-label` nos chips,
   `role="status"` no toast, `aria-live` no resultado, `aria-busy` durante a
   análise, skip link, `:focus-visible`, `prefers-reduced-motion`, tabelas com
   scroll horizontal no mobile.
8. **Links reais.** CTA, rodapé e nav apontam para HF/GitHub/LinkedIn em vez
   de `#`.

### Execução local

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860
pytest -q                # 18 testes, sem GPU (runtime falso)
```

Variáveis: `OPF_DEVICE=cuda|cpu`, `OPF_EAGER_LOAD=0` (não baixa o modelo),
`HF_TOKEN`.

### Privacidade e limitações

O texto enviado é processado pelo Space hospedado. Não envie dados confidenciais ou pessoais reais sem avaliar a infraestrutura, os controles de acesso e as políticas aplicáveis. A detecção pode produzir falsos positivos e deixar dados passar; revise o resultado antes de compartilhar ou usar para redação. As métricas dependem dos conjuntos de avaliação e não garantem desempenho em outros domínios.

---

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
| [gliner-ettin-68m-ptbr-pii-full-3x-v1-checkpoints/gliner-ettin-68m-ptbr-pii-top50k-v1](https://huggingface.co/arthrod/gliner-ettin-68m-ptbr-pii-full-3x-v1-checkpoints/tree/main/gliner-ettin-68m-ptbr-pii-top50k-v1) | model | GLiNER on `jhu-clsp/ettin-encoder-68m`, 50k curated subset, 11,700 steps. Ablation: shows the curated subset under-generalizes. |
| [arthrod/gliner-mmbert-small-ptbr-pii-full-3x-v1](https://huggingface.co/arthrod/gliner-mmbert-small-ptbr-pii-full-3x-v1) | model | GLiNER on `jhu-clsp/mmBERT-small`, full train set + 2k Ritápolis rows, 69,000 steps. Strongest GLiNER model of the family. |
| [arthrod/gliner-opf-ptbr-pii-v1](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1) | model | Fine-tune of OpenAI Privacy Filter (not GLiNER despite the name). Best partial F1 on the in-distribution val set and by far the fewest false positives on no-PII text. |
| [arthrod/gliner-opf-ptbr-pii-bench-v1](https://huggingface.co/datasets/arthrod/gliner-opf-ptbr-pii-bench-v1) | dataset | Benchmark: predictions and metrics for the mmBERT-small and Privacy Filter models (plus two unreleased ettin variants) on the PT-BR val set, held-out sources and no-PII spam/phishing text. |
| **arthrod/gliner-opf-ptbr-pii-demo** (this repo) | space | Gradio demo running the Privacy Filter fine-tune. |
| `arthrod/gliner_eval_folder` (private) | dataset | Earlier (Feb 2026) sweep: 18 GLiNER models x 9 eval sets. |
| [arthrod/gliner_review_comparison](https://huggingface.co/datasets/arthrod/gliner_review_comparison) | dataset | Earlier GLiNER PII review/comparison data. |
