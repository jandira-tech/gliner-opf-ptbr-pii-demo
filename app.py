"""OPF PT-BR PII demo — fine-tuned openai/privacy-filter on 914k PT-BR samples."""
from __future__ import annotations
# spaces FIRST so its CUDA emulation hooks before torch / opf load
import spaces  # noqa: F401
import os, hashlib, html
from typing import List

import torch
import gradio as gr
from huggingface_hub import snapshot_download
from opf import OPF

REPO_ID = "arthrod/gliner-opf-ptbr-pii-v1"

# 24 PT-BR canonical labels (the comparison set vs GLiNER models)
PTBR_LABELS = [
    "cpf_document_number", "rg_document_number", "pis_document_number",
    "credit_card", "phone_number", "email_address",
    "first_name", "middle_name", "last_name",
    "dob",
    "location_street", "location_building_number", "location_neighborhood",
    "location_city", "location_state", "location_state_abbreviation",
    "location_zip", "location_full_address",
    "personal_description_of_ethnicity",
    "personal_description_of_medical_conditions",
    "personal_description_of_organizational_affiliation",
    "personal_description_of_political_opinion",
    "personal_description_of_religious_convictions",
    "personal_description_of_sexual_information",
]

# Extras the model also knows (47 cross-source labels) — surfaced for advanced users
EXTRA_LABELS = [
    "date", "company_name", "time", "id_card_number", "user_name", "url",
    "title", "occupation", "country", "age", "account_number",
    "social_security_number", "sex_or_gender", "tax_id_number",
    "password", "driver_license_number", "customer_id", "passport_number",
    "date_time", "employee_id", "education_level", "biometric_identifier",
    "vehicle_identifier", "coordinate", "religious_belief", "employment_status",
    "ipv4", "political_view", "certificate_license_number", "medical_record_number",
    "fax_number", "language", "county", "license_plate",
    "health_plan_beneficiary_number", "bank_routing_number", "pin",
    "mac_address", "swift_bic", "api_key", "http_cookie", "blood_type",
    "device_identifier", "ipv6", "cvv", "sexuality", "unique_id",
]

ALL_LABELS = PTBR_LABELS + EXTRA_LABELS

DESCRIPTION = """
# 🛡️ OPF PT-BR PII Detector

Fine-tune do [**openai/privacy-filter**](https://github.com/openai/privacy-filter) em **914 mil amostras de PII** em português brasileiro (texto natural das fontes originais — sem augmentation artificial).

**73 labels** suportados (24 PT-BR canônicos + 47 cross-source multilíngua). Decoder Viterbi com restrições BIESO.

> Comparação: este modelo bate o **mmBERT-small × 3** em **partial F1 (0.897 vs 0.888)** e perde por pouco em **exact F1 (0.853 vs 0.870)**. Detalhes no [model card](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1).
"""

def _label_color(label: str) -> str:
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
    r = 100 + (r % 130); g = 100 + (g % 130); b = 100 + (b % 130)
    return f"#{r:02x}{g:02x}{b:02x}"


# Download our private model checkpoint at startup. ZeroGPU's CUDA emulation
# (active via `import spaces`) lets us touch CUDA-pinned weights here.
_HF_TOKEN = os.getenv("HF_TOKEN")
print(f"Downloading {REPO_ID} ...")
_LOCAL_PATH = snapshot_download(REPO_ID, token=_HF_TOKEN, allow_patterns=[
    "config.json", "model.safetensors", "USAGE.txt", "finetune_summary.json",
])
print(f"  -> {_LOCAL_PATH}")

# Load the OPF runtime once
_RUNTIME = OPF(model=_LOCAL_PATH, device="cuda", output_mode="typed", output_text_only=False)
print("OPF runtime ready")


@spaces.GPU
def predict(text: str, labels_filter: List[str], discard_overlapping: bool):
    if not text or not text.strip():
        return gr.update(value=[(text or "", None)], color_map={}), "_(sem texto)_"

    label_set = set(labels_filter) if labels_filter else None  # None = no filter

    result = _RUNTIME.redact(text)
    spans = list(result.detected_spans)
    if label_set is not None:
        spans = [s for s in spans if s.label in label_set]
    spans.sort(key=lambda s: s.start)

    # Build HighlightedText segments
    segments = []
    cursor = 0
    for s in spans:
        if s.start < cursor:
            continue   # overlap, skip when not multi-label
        if cursor < s.start:
            segments.append((text[cursor:s.start], None))
        segments.append((text[s.start:s.end], s.label))
        cursor = s.end
    if cursor < len(text):
        segments.append((text[cursor:], None))

    found = sorted({s.label for s in spans})
    color_map = {l: _label_color(l) for l in found}

    header_bits = [f"**{len(spans)} entidades detectadas**"]
    if label_set is not None:
        header_bits.append(f"filtro: {len(label_set)} labels")
    if discard_overlapping:
        header_bits.append("overlap: discarded")
    header = " · ".join(header_bits)

    if not spans:
        summary = f"{header}\n\n_(nenhuma entidade encontrada)_"
    else:
        rows = ["| Label | Span | Placeholder |", "|---|---|---|"]
        for s in spans:
            rows.append(f"| `{s.label}` | `{html.escape(s.text)}` | `{html.escape(s.placeholder)}` |")
        summary = header + "\n\n" + "\n".join(rows)
    return gr.update(value=segments, color_map=color_map), summary


# Hand-crafted PT-BR examples — same family as the GLiNER demo
EXAMPLES = [
    ["Galera, esqueci a senha do wifi de novo 😅 alguém lembra? E aproveitando: meu novo número é (11) 98432-7765, mudei de operadora ontem. Beijo!", PTBR_LABELS, False],
    ["Mãe, cheguei em casa! O motorista do Uber foi gente boa, plaquinha QYV-4538. Pago na próxima visita, prometo. Te amo 💚", PTBR_LABELS + ["license_plate"], False],
    ["Prezada Mariana Cordeiro de Almeida,\n\nConforme conversado, segue oferta para a vaga de Engenheira de Dados Pleno. Salário: R$ 14.500,00 + benefícios. Início previsto: 15/06/2026. Por favor, envie CPF, RG e comprovante de residência para mariana.almeida@candidato.com.br até sexta. Qualquer dúvida, ligue (11) 3045-7788.\n\nAtt., Bruno Sales — Talent Acquisition", PTBR_LABELS + ["company_name", "occupation"], False],
    ["Cobrança em aberto: cliente Augusto Pereira da Silveira, CPF 412.789.345-22, contrato #BR-2024-09887. Valor: R$ 2.347,90. Vencimento original: 12/03/2026. Para regularizar, ligue 0800-722-3344 ou envie comprovante para cobranca@bancosp.com.br.", PTBR_LABELS + ["account_number"], False],
    ["EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 4ª VARA CÍVEL DE BELO HORIZONTE/MG\n\nMARIA DE LOURDES SANT'ANA FERREIRA, brasileira, viúva, professora aposentada, portadora do RG MG-3.456.789 SSP/MG, inscrita no CPF sob nº 087.654.321-90, residente e domiciliada à Rua Padre Eustáquio, nº 2.115, apto 802, bairro Padre Eustáquio, CEP 30730-130, vem, respeitosamente, por seu advogado infra-assinado (Dr. Ricardo Camargo Pinto, OAB/MG 145.789), propor a presente AÇÃO DE INDENIZAÇÃO POR DANOS MORAIS em face de TIM CELULAR S.A.", PTBR_LABELS + ["company_name"], False],
    ["Paciente: Otávio Fernandes Coutinho, 58 anos, masculino, casado, comerciante, natural de Recife/PE, residente em Olinda/PE.\nQueixa principal: dor torácica há 3 dias.\nHistórico: hipertenso há 12 anos (uso de Losartana 50mg), diabético tipo 2 (Metformina 850mg 2x/dia), tabagista (35 maços/ano), católico apostólico romano (recusa transfusão por motivos pessoais — anotação de enfermagem).", PTBR_LABELS + ["age", "occupation"], False],
    ["CURRÍCULO\n\nFernanda Caroline Bittencourt de Andrade\nNascida em 14/07/1991, brasileira, solteira\n📍 Rua Joaquim Floriano, 820, ap. 134 — Itaim Bibi, São Paulo/SP — CEP 04534-002\n📱 (11) 97654-1122 | ✉ fer.bittencourt91@gmail.com\nLinkedIn: /in/fernandabittencourt\n\nFORMAÇÃO\n• Mestrado em Ciência de Dados — USP (2020-2022)\n• Bacharel em Estatística — UFMG (2010-2014)\n\nEXPERIÊNCIA\n• Sr. Data Scientist @ Nubank (2023-atual)\n• Data Scientist @ iFood (2018-2023)", PTBR_LABELS + ["company_name", "education_level", "occupation", "url"], False],
    ["GENTE PELO AMOR DE DEUS 😭 perdi minha carteira no metrô da estação Sé hoje por volta das 18h30. Continha:\n• CPF 234.567.890-12\n• RG 11.222.333-4 SSP/SP\n• CNH 04567823901\n• Cartão Bradesco final 5678\n• Cartão Nubank final 2244\n• Vale-transporte e R$ 180 em dinheiro\n• Foto da minha avó (essa que dói)\n\nQuem achar, favor ligar (11) 99999-1111 ou mandar pra Avenida Angélica, 2530, apto 91, Higienópolis. Recompenso!! 🙏\n— Tatiana Vasconcelos", PTBR_LABELS + ["driver_license_number"], False],
    ["Apresento minha família: meu pai Waldisnei Mengão Pereira, minha mãe Waldisneia da Conceição Mengão (sim, casaram porque os nomes combinavam 😅), meus irmãos Wellington Kennedy Mengão, Donizete Bombril Mengão e Xerox Maradona Mengão. Já minha irmã caçula é a Disney Princess Mengão Pereira — homenagem dos pais aos sonhos americanos. Moramos todos em Cariacica, ES.", PTBR_LABELS, False],
    ["Lista de presença da turma 5°B - EE Tancredo Neves:\n• Jhonatan Felipy Albuquerque Tomé\n• Maria Aparecida Tiradentes Costa\n• Luiz Felipe Cristiano Ronaldo Silva\n• Karoline Beyoncé Vasconcellos\n• Yuri Gagarin Nascimento Junior\n• Cleydiomar Bonifácio Pereira Filho\n• Anny Cleyciane Vitória Souza\n• Brendow Wesley Marques\n• Gleicianne Aparecida da Silveira\n• Jhonny Tony Caetano Veloso (sim, esse mesmo)\nProf. responsável: Maria Goretti dos Santos — (31) 98712-3344", PTBR_LABELS, False],
    ["Histórico psiquiátrico: paciente Camila Verçosa Bandeira, 34, transtorno bipolar tipo II, em tratamento há 7 anos com Lítio 600mg/dia + Lamotrigina 200mg. Episódio depressivo grave em out/2024 com tentativa de autoextermínio (overdose de benzodiazepínicos). Acompanhamento semanal com psicóloga (Dra. Helena Vasconcelos, CRP 06/89745).", PTBR_LABELS + ["age"], False],
    ["BOLETIM DE OCORRÊNCIA Nº 2026/0044521\nNatureza: Lesão corporal culposa em acidente de trânsito\nData/hora: 22/04/2026 às 14h17\nLocal: cruzamento da Rua Augusta com Rua Frei Caneca, Consolação, São Paulo/SP\nVítima 1: Letícia Albuquerque Cunha, 28 anos, brasileira, designer, RG 47.892.301-X SSP/SP, CPF 567.890.123-45, residente à Rua Bela Cintra, 1990, ap. 53, Consolação, SP — telefone (11) 96677-1144\nCondutor responsável: Hugo Leandro Pinto, 41 anos, motorista app, CNH 04772819220, placa do veículo BRA-2E37 (Honda Civic preto)", PTBR_LABELS + ["age", "license_plate", "driver_license_number", "occupation"], False],
    ["Hi team, this is Joana — joana.alves@globalcorp.com — based in São Paulo. Heads up: our client François Dubois (CEO of LightSpeed Paris, +33 6 12 34 56 78) will be in Brazil next week visiting our office at Av. Brigadeiro Faria Lima, 3500, 12º andar, Itaim Bibi. Please make sure his hotel near Rua Oscar Freire is booked. Thx!", PTBR_LABELS + ["company_name", "title"], False],
]


_CSS = """
#examples-block .table-wrap td:first-child,
#examples-block .table-wrap th:first-child {
    max-width: 800px; min-width: 480px;
    white-space: pre-wrap !important;
    word-break: break-word;
}
#examples-block .table-wrap td { vertical-align: top; padding: 8px 10px; }
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red"),
               title="OPF PT-BR PII", css=_CSS) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(label="Texto", placeholder="Cole um trecho com possível PII...", lines=8)
        with gr.Column(scale=1):
            discard_overlapping = gr.Checkbox(
                label="Descartar spans sobrepostos do mesmo label",
                value=False,
            )

    labels_filter = gr.Dropdown(
        choices=ALL_LABELS,
        value=PTBR_LABELS,
        label="Labels (deixe vazio para usar TODOS os 73 labels do modelo)",
        multiselect=True,
    )

    btn = gr.Button("🛡️ Detectar PII", variant="primary")

    output_highlight = gr.HighlightedText(label="Texto anotado", combine_adjacent=False, show_legend=True)
    output_summary = gr.Markdown(label="Predições")

    btn.click(predict, [input_text, labels_filter, discard_overlapping], [output_highlight, output_summary])

    with gr.Group(elem_id="examples-block"):
        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text, labels_filter, discard_overlapping],
            outputs=[output_highlight, output_summary],
            fn=predict,
            cache_examples=False,
            examples_per_page=6,
            label="📋 Exemplos prontos (criados à mão, NÃO vêm do treino)",
        )

    gr.Markdown("""
    ---
    ### Sobre

    | Item | Detalhe |
    |---|---|
    | Modelo | `arthrod/gliner-opf-ptbr-pii-v1` (private) |
    | Backbone | openai/privacy-filter (8L sparse-MoE, 128 experts) |
    | Treino | 9-round chunked sweep (3 ép × 3 saves), 914k rows, ~3h45m wall em MI300X |
    | Decoder | Constrained Viterbi (BIESO) |
    | Best ckpt | e3_c3 — typed F1 0.885 (P 0.894 / R 0.876) |

    **Diferenças vs GLiNER (mmBERT/ettin):**
    - Vocabulário 73 labels vs 24 (inclui SSN, passport, IP, MAC, swift_bic, etc.)
    - Inferência via Viterbi constrained vs token-level argmax
    - Sem easter-egg `berco-de-tiradentes` (use o demo mmBERT pra essa)

    Ranking apples-to-apples (mesma val, mesmas 24 labels PT-BR, nervaluate):

    | Modelo | partial F1 | exact F1 |
    |---|---|---|
    | **opf-finetune (este)** | **0.897** | 0.853 |
    | mmBERT-small × 3 | 0.888 | **0.870** |
    | ettin-68m | 0.822 | 0.789 |
    | ettin-32m | 0.808 | 0.769 |
    """)


if __name__ == "__main__":
    demo.queue(max_size=10).launch()
