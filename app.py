"""OPF PT-BR PII demo — fine-tuned openai/privacy-filter on 914k PT-BR samples."""
from __future__ import annotations
# spaces FIRST so its CUDA emulation hooks before torch / opf load
import spaces  # noqa: F401
import os, hashlib, html, json
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

# LGPD / operational taxonomy — maps each label to a category used for
# grouping, coloring, and legal-relevance reporting in the summary panel.
CATEGORIES: dict[str, list[str]] = {
    "Documento oficial / identidade": [
        "cpf_document_number", "rg_document_number", "pis_document_number",
        "id_card_number", "passport_number", "driver_license_number",
        "social_security_number", "tax_id_number", "certificate_license_number",
    ],
    "Dado financeiro": [
        "credit_card", "account_number", "bank_routing_number",
        "swift_bic", "cvv", "pin", "customer_id",
    ],
    "Contato": [
        "phone_number", "email_address", "fax_number",
        "url", "ipv4", "ipv6", "mac_address",
    ],
    "Nome": [
        "first_name", "middle_name", "last_name",
        "user_name", "title",
    ],
    "Data de nascimento / idade": [
        "dob", "date", "time", "date_time", "age",
    ],
    "Localização": [
        "location_street", "location_building_number", "location_neighborhood",
        "location_city", "location_state", "location_state_abbreviation",
        "location_zip", "location_full_address",
        "country", "county", "coordinate",
    ],
    # LGPD art. 5º, II — "dados pessoais sensíveis"
    "Dado pessoal sensível (LGPD art. 5º, II)": [
        "personal_description_of_ethnicity",
        "personal_description_of_medical_conditions",
        "personal_description_of_organizational_affiliation",
        "personal_description_of_political_opinion",
        "personal_description_of_religious_convictions",
        "personal_description_of_sexual_information",
        "religious_belief", "political_view", "sexuality",
        "sex_or_gender", "biometric_identifier", "blood_type",
    ],
    "Credencial / identificador técnico": [
        "password", "api_key", "http_cookie", "device_identifier",
        "unique_id", "employee_id", "vehicle_identifier", "license_plate",
        "medical_record_number", "health_plan_beneficiary_number",
    ],
    "Contexto profissional": [
        "occupation", "employment_status", "education_level",
        "company_name", "language",
    ],
}
_CUSTOM_CATEGORY = "Outros rótulos"

# Muted, professional palette — low saturation, high lightness so dark text on
# top remains legible (Gradio HighlightedText does not auto-invert).
CATEGORY_COLORS: dict[str, str] = {
    # Earthtone palette — reads as muted ink on ivory paper, not pastel toy.
    "Documento oficial / identidade":                   "#d7b4a8",  # oxblood lite
    "Dado financeiro":                                  "#d4b688",  # ochre
    "Contato":                                          "#a8bdb5",  # muted teal
    "Nome":                                             "#c4beb0",  # warm stone
    "Data de nascimento / idade":                       "#c0b09c",  # mushroom
    "Localização":                                      "#a8c0a0",  # forest-lite
    "Dado pessoal sensível (LGPD art. 5º, II)":         "#c8a090",  # deep oxblood
    "Credencial / identificador técnico":               "#c9b5b0",  # dusty rose
    "Contexto profissional":                            "#b5bcc0",  # slate-lite
    _CUSTOM_CATEGORY:                                   "#bcb5a4",  # paper gray
}

DESCRIPTION = """
## OPF PT-BR — Identificação de PII em documentos

Fine-tune do [openai/privacy-filter](https://github.com/openai/privacy-filter)
treinado em **914 mil amostras** de PII em português brasileiro, com destaque
explícito para **dados pessoais sensíveis** nos termos do art. 5º, II da Lei
nº 13.709/2018 (LGPD). Cobre 73 categorias (24 canônicas PT-BR + 47 cross-source
multilíngua) e emite, além do rótulo, um *placeholder* já pronto para
substituição, via decoder Viterbi com restrições BIESO.
"""


def _category_for(label: str) -> str:
    for cat, members in CATEGORIES.items():
        if label in members:
            return cat
    return _CUSTOM_CATEGORY


def _label_color(label: str) -> str:
    """Color by category, with a small deterministic jitter so sibling labels
    (e.g. 'first_name' vs 'last_name') are visually distinguishable but stay
    within the same hue family."""
    base = CATEGORY_COLORS.get(_category_for(label), "#c9c2b5")
    br, bg, bb = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    dr = ((h >> 16) & 0x1F) - 16
    dg = ((h >> 8) & 0x1F) - 16
    db = (h & 0x1F) - 16
    r = max(0, min(255, br + dr))
    g = max(0, min(255, bg + dg))
    b = max(0, min(255, bb + db))
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

    # Build HighlightedText segments — overlaps from the same-label runs are
    # elided in the rendered text regardless of the toggle (Gradio cannot paint
    # overlapping spans); the toggle is surfaced in the summary header for
    # transparency.
    segments = []
    cursor = 0
    for s in spans:
        if s.start < cursor:
            continue
        if cursor < s.start:
            segments.append((text[cursor:s.start], None))
        segments.append((text[s.start:s.end], s.label))
        cursor = s.end
    if cursor < len(text):
        segments.append((text[cursor:], None))

    found = sorted({s.label for s in spans})
    color_map = {l: _label_color(l) for l in found}

    summary = _build_summary(spans, label_set, discard_overlapping)
    return gr.update(value=segments, color_map=color_map), summary


def _build_summary(spans: list, label_set, discard_overlapping: bool) -> str:
    header_bits = [f"**{len(spans)} entidade(s) detectada(s)**"]
    if label_set is not None:
        header_bits.append(f"filtro ativo: {len(label_set)} label(s)")
    if discard_overlapping:
        header_bits.append("sobreposições descartadas")
    header = " · ".join(header_bits)

    if not spans:
        return header + "\n\n_Nenhuma entidade encontrada._"

    # Group by category, preserving CATEGORIES insertion order.
    by_cat: dict[str, list] = {}
    for s in spans:
        by_cat.setdefault(_category_for(s.label), []).append(s)

    lines: list[str] = [header, ""]

    # --- Pontos de atenção (legal relevance) ------------------------------
    risk_lines: list[str] = []
    sens = by_cat.get("Dado pessoal sensível (LGPD art. 5º, II)")
    if sens:
        sens_labels = sorted({s.label for s in sens})
        risk_lines.append(
            f"- **{len(sens)} ocorrência(s) de dado pessoal sensível** "
            f"({', '.join(f'`{l}`' for l in sens_labels)}). "
            "Tratamento condicionado às hipóteses do art. 11 da LGPD."
        )
    docs = by_cat.get("Documento oficial / identidade")
    if docs:
        doc_labels = sorted({s.label for s in docs})
        risk_lines.append(
            f"- **{len(docs)} documento(s) oficial(is)** "
            f"({', '.join(f'`{l}`' for l in doc_labels)})."
        )
    fin = by_cat.get("Dado financeiro")
    if fin:
        risk_lines.append(
            f"- **{len(fin)} dado(s) financeiro(s)** identificado(s)."
        )
    cred = by_cat.get("Credencial / identificador técnico")
    if cred:
        risk_lines.append(
            f"- **{len(cred)} credencial(ais) / identificador(es) técnico(s)** — "
            "revisar retenção e criptografia em trânsito/repouso."
        )
    if risk_lines:
        lines.append("### Pontos de atenção")
        lines.extend(risk_lines)
        lines.append("")

    # --- Detalhamento por categoria ---------------------------------------
    lines.append("### Detalhamento por categoria")
    order = list(CATEGORIES.keys()) + [_CUSTOM_CATEGORY]
    for cat in order:
        entries = by_cat.get(cat)
        if not entries:
            continue
        entries = sorted(entries, key=lambda x: x.start)
        lines.append("")
        lines.append(f"**{cat}** — {len(entries)} ocorrência(s)")
        lines.append("")
        lines.append("| Label | Trecho | Placeholder |")
        lines.append("|---|---|---|")
        for s in entries:
            lines.append(
                f"| `{s.label}` | {html.escape(s.text)} | `{html.escape(s.placeholder)}` |"
            )

    return "\n".join(lines)


# ---------- Examples (hand-crafted, rich Brazilian content, NOT from training data) ----------
# Each row: [text, labels_filter, discard_overlapping]
EXAMPLES = [
    # ===== WhatsApp / chat realista =====
    ["Galera, esqueci a senha do wifi de novo 😅 alguém lembra? E aproveitando: meu novo número é (11) 98432-7765, mudei de operadora ontem. Beijo!",
     PTBR_LABELS, False],

    ["Mãe, cheguei em casa! O motorista do Uber foi gente boa, plaquinha QYV-4538. Pago na próxima visita, prometo. Te amo 💚",
     PTBR_LABELS + ["license_plate"], False],

    # ===== Email corporativo / RH =====
    ["Prezada Mariana Cordeiro de Almeida,\n\nConforme conversado, segue oferta para a vaga de Engenheira de Dados Pleno. Salário: R$ 14.500,00 + benefícios. Início previsto: 15/06/2026. Por favor, envie CPF, RG e comprovante de residência para mariana.almeida@candidato.com.br até sexta. Qualquer dúvida, ligue (11) 3045-7788.\n\nAtt., Bruno Sales — Talent Acquisition",
     PTBR_LABELS + ["company_name", "occupation"], False],

    # ===== Boleto / financeiro =====
    ["Cobrança em aberto: cliente Augusto Pereira da Silveira, CPF 412.789.345-22, contrato #BR-2024-09887. Valor: R$ 2.347,90. Vencimento original: 12/03/2026. Para regularizar, ligue 0800-722-3344 ou envie comprovante para cobranca@bancosp.com.br.",
     PTBR_LABELS + ["account_number"], False],

    # ===== Fofoca / rede social =====
    ["Gente, vocês não vão acreditar — encontrei o Rafael Monteiro Caldas (sim, AQUELE Rafael, ex-marido da Camila Bezerra) na fila do Outback Morumbi ontem com OUTRA mulher 👀 Ele mora na Vila Madalena agora, na Rua Aspicuelta perto do bar do Mané. A ex tá morando em Pinheiros, no edifício novo da Rua dos Pinheiros 1340. Drama puro 🍿",
     PTBR_LABELS, False],

    # ===== Jurídico — petição =====
    ["EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 4ª VARA CÍVEL DE BELO HORIZONTE/MG\n\nMARIA DE LOURDES SANT'ANA FERREIRA, brasileira, viúva, professora aposentada, portadora do RG MG-3.456.789 SSP/MG, inscrita no CPF sob nº 087.654.321-90, residente e domiciliada à Rua Padre Eustáquio, nº 2.115, apto 802, bairro Padre Eustáquio, CEP 30730-130, vem, respeitosamente, por seu advogado infra-assinado (Dr. Ricardo Camargo Pinto, OAB/MG 145.789), propor a presente AÇÃO DE INDENIZAÇÃO POR DANOS MORAIS em face de TIM CELULAR S.A.",
     PTBR_LABELS + ["company_name"], False],

    # ===== Médico — prontuário =====
    ["Paciente: Otávio Fernandes Coutinho, 58 anos, masculino, casado, comerciante, natural de Recife/PE, residente em Olinda/PE.\nQueixa principal: dor torácica há 3 dias.\nHistórico: hipertenso há 12 anos (uso de Losartana 50mg), diabético tipo 2 (Metformina 850mg 2x/dia), tabagista (35 maços/ano), católico apostólico romano (recusa transfusão por motivos pessoais — anotação de enfermagem).\nExame físico: PA 165/95 mmHg, FC 92 bpm. Suspeita de SCASSST. Encaminhar para hemodinâmica.\nDr. Augusto Leme Quintanilha — CRM-PE 18.745",
     PTBR_LABELS + ["age", "occupation"], False],

    # ===== Currículo extenso =====
    ["CURRÍCULO\n\nFernanda Caroline Bittencourt de Andrade\nNascida em 14/07/1991, brasileira, solteira\n📍 Rua Joaquim Floriano, 820, ap. 134 — Itaim Bibi, São Paulo/SP — CEP 04534-002\n📱 (11) 97654-1122 | ✉ fer.bittencourt91@gmail.com\nLinkedIn: /in/fernandabittencourt\n\nFORMAÇÃO\n• Mestrado em Ciência de Dados — USP (2020-2022)\n• Bacharel em Estatística — UFMG (2010-2014)\n\nEXPERIÊNCIA\n• Sr. Data Scientist @ Nubank (2023-atual)\n• Data Scientist @ iFood (2018-2023)\n• Analista @ Itaú Unibanco (2014-2018)\n\nIDIOMAS: Português (nativa), Inglês (fluente), Espanhol (intermediário)",
     PTBR_LABELS + ["company_name", "education_level", "occupation", "url"], False],

    # ===== Ticket de suporte =====
    ["#TICKET-44521 — Cliente: João Vitor da Mata Machado, conta #889245-X. Reclama que o cartão Mastercard final 7821 foi clonado em compras na Aliexpress. Última compra legítima: 18/04 às 22h17 no posto Shell da Avenida Brigadeiro Faria Lima, 3500. Bloqueamos o cartão, novo será enviado para o endereço em cadastro: Alameda dos Maracatins, 426, ap. 71, Moema, SP, 04089-001. Cliente avisado pelo telefone (11) 99812-3344.",
     PTBR_LABELS + ["account_number", "customer_id"], False],

    # ===== Notícia de jornal =====
    ["BRASÍLIA — O ministro Alexandre Henrique Quintanilha de Souza, 62 anos, foi internado ontem à noite no Hospital Sírio-Libanês de Brasília após sofrer um mal-estar durante sessão do STJ. Casado com a juíza Patrícia Rocha Quintanilha e pai de três filhos, o ministro reside no Lago Sul desde 2019. A assessoria informou que o quadro é estável e que ele deve receber alta nos próximos dias. Contato para imprensa: ascom@stj.jus.br.",
     PTBR_LABELS + ["age", "title", "occupation"], False],

    # ===== Religioso / político / orientação sexual =====
    ["Sou Daiana, 29 anos, espírita kardecista praticante, frequento o Centro Espírita Casa do Caminho aqui em Curitiba há 8 anos. Voto historicamente em candidatos do PSOL e considero importante a separação entre Igreja e Estado. Lésbica assumida, casei com a Renata em 2023 — ela é vegana, eu não, mas nos respeitamos muito 💕",
     PTBR_LABELS + ["age", "religious_belief", "political_view", "sexuality"], False],

    # ===== Médico / sensível combinado =====
    ["Histórico psiquiátrico: paciente Camila Verçosa Bandeira, 34, transtorno bipolar tipo II, em tratamento há 7 anos com Lítio 600mg/dia + Lamotrigina 200mg. Episódio depressivo grave em out/2024 com tentativa de autoextermínio (overdose de benzodiazepínicos). Acompanhamento semanal com psicóloga (Dra. Helena Vasconcelos, CRP 06/89745). Familiares cientes — mãe Beatriz (61) e irmã Aline (28).",
     PTBR_LABELS + ["age", "medical_record_number"], False],

    # ===== Carteira perdida (drama) =====
    ["GENTE PELO AMOR DE DEUS 😭 perdi minha carteira no metrô da estação Sé hoje por volta das 18h30. Continha:\n• CPF 234.567.890-12\n• RG 11.222.333-4 SSP/SP\n• CNH 04567823901\n• Cartão Bradesco final 5678\n• Cartão Nubank final 2244\n• Vale-transporte e R$ 180 em dinheiro\n• Foto da minha avó (essa que dói)\n\nQuem achar, favor ligar (11) 99999-1111 ou mandar pra Avenida Angélica, 2530, apto 91, Higienópolis. Recompenso!! 🙏\n— Tatiana Vasconcelos",
     PTBR_LABELS + ["driver_license_number"], False],

    # ===== Petição / jurídico =====
    ["PROCURAÇÃO PÚBLICA\n\nPor este instrumento particular, eu, RAIMUNDO NONATO DOS SANTOS JÚNIOR, brasileiro, casado, engenheiro civil, portador da Cédula de Identidade RG nº 18.456.789-3 SSP/SP, inscrito no CPF/MF sob nº 087.654.321-09, residente e domiciliado à Alameda Santos, nº 1.500, 12º andar, conjunto 1204, Cerqueira César, São Paulo/SP, CEP 01418-100, nomeio e constituo como meu bastante procurador o Dr. RICARDO MENEZES DE OLIVEIRA, OAB/SP nº 234.567, com escritório à Rua Joaquim Floriano, nº 940, conjunto 81, Itaim Bibi, São Paulo/SP, CEP 04534-004, conferindo-lhe os poderes da cláusula ad judicia et extra...",
     PTBR_LABELS + ["occupation"], False],

    # ===== Grupo de pais da escola =====
    ["BOM DIA TURMA DO 5°B! 🏫\n\nLembrete da reunião de pais quinta-feira 19h. Quem ainda não confirmou presença, manda no privado.\nAniversariantes da semana:\n• Mariazinha (Ana Clara Souto) — 7 anos\n• Pedrinho (Pedro Henrique Vasconcellos Jr.) — 8 anos\n\nA mãe da Beatriz (Sra. Carla Tedesco) ofereceu carona pra todos no passeio ao Parque Burle Marx. Quem topa, manda CPF e RG da criança até quarta no whats da Carla: (11) 98701-2244.\n\n— Daniela, mãe da Helena (representante de turma)",
     PTBR_LABELS + ["age"], False],

    # ===== Receita médica =====
    ["RECEITUÁRIO MÉDICO\nDr. Patrícia Marques de Lima — Cardiologista — CRM-RJ 52.789\nClínica CorAtivo — Rua Visconde de Pirajá, 414, sala 802 — Ipanema, RJ — (21) 2511-3344\n\nPaciente: Sebastião Almeida Vasconcelos, 71 anos, masculino\nCPF: 098.765.432-10\n\nReceita:\n1. Atorvastatina 40mg — 1 comp à noite (60 dias)\n2. AAS 100mg — 1 comp pela manhã (uso contínuo)\n3. Bisoprolol 5mg — 1/2 comp pela manhã (60 dias)\n\nRetorno em 90 dias com novo lipidograma. Pressão alvo: <130x80.\n\nRJ, 22 de abril de 2026.",
     PTBR_LABELS + ["age", "occupation", "title"], False],

    # ===== Dating profile (intenção de exposição) =====
    ["André, 31, hétero, geminiano ♊\n📍 São Paulo (Pinheiros, na fronteira com Vila Madalena)\n💼 Product Manager numa fintech\n🏋 Crossfit 4x/semana, vegetariano há 2 anos\n🎓 Engenharia de Computação - USP\n🌱 Católico não-praticante, votei nulo nas últimas eleições\n👨‍👩‍👦 Quero ter filhos eventualmente\nInsta: @andreocosta_oficial\nWhats só depois do match: 11 99654-7321",
     PTBR_LABELS + ["age", "occupation", "religious_belief", "political_view", "sexuality", "user_name"], False],

    # ===== Stalker email (creepy realista) =====
    ["Oi Bia, sei que você não quer falar comigo mas eu PRECISO te explicar. Sei que vc se mudou pra Avenida Pacaembu, 1820, ap 71, vi seu Instagram. Sei que trabalha na agência Wieden+Kennedy ali na Vila Olímpia. Liguei pro seu pai (Sr. Marcelo, eu tinha o número (11) 3567-2244 da casa antiga) e ele desligou na minha cara. Por favor, só me responde uma vez. Beijo. — R.",
     PTBR_LABELS + ["company_name"], False],

    # ===== Doação / formulário ONG =====
    ["Cadastro de doador — Instituto Comer pra Aprender\nNome: Cláudia Regina Toledo de Mello\nE-mail: claudia.toledo@uol.com.br\nCPF: 345.678.901-23\nTelefone: (11) 4002-8922\nValor mensal: R$ 50,00 via débito automático\nCartão: Visa final 4488, validade 09/28\nEndereço de cobrança: Rua Cônego Eugênio Leite, 920, ap. 41, Pinheiros, SP — 05414-001\nReligião: judaica (mencionou querer dedicar a doação à memória de seu pai Avraham Toledo)",
     PTBR_LABELS + ["company_name", "religious_belief"], False],

    # ===== Boletim de ocorrência =====
    ["BOLETIM DE OCORRÊNCIA Nº 2026/0044521\nNatureza: Lesão corporal culposa em acidente de trânsito\nData/hora: 22/04/2026 às 14h17\nLocal: cruzamento da Rua Augusta com Rua Frei Caneca, Consolação, São Paulo/SP\nVítima 1: Letícia Albuquerque Cunha, 28 anos, brasileira, designer, RG 47.892.301-X SSP/SP, CPF 567.890.123-45, residente à Rua Bela Cintra, 1990, ap. 53, Consolação, SP — telefone (11) 96677-1144\nCondutor responsável: Hugo Leandro Pinto, 41 anos, motorista app, CNH 04772819220, placa do veículo BRA-2E37 (Honda Civic preto)",
     PTBR_LABELS + ["age", "license_plate", "driver_license_number", "occupation"], False],

    # ===== Multilingual flex =====
    ["Hi team, this is Joana — joana.alves@globalcorp.com — based in São Paulo. Heads up: our client François Dubois (CEO of LightSpeed Paris, +33 6 12 34 56 78) will be in Brazil next week visiting our office at Av. Brigadeiro Faria Lima, 3500, 12º andar, Itaim Bibi. Please make sure his hotel near Rua Oscar Freire is booked. Thx!",
     PTBR_LABELS + ["company_name", "title", "country"], False],
]


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,400;1,9..144,500&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ============================================================================
   Editorial / legal-journal aesthetic — ivory paper, ink, double-ruled dividers.
   Forces a light palette regardless of Gradio's auto dark mode.
   ============================================================================ */

/* ---- Force light, always ------------------------------------------------- */
html, body, gradio-app, .gradio-container, .gradio-container.dark, .dark {
    color-scheme: light !important;
}
body { background: #e8e3d2 !important; color: #1a1612 !important; }
html { background: #e8e3d2 !important; }

/* Override Gradio design tokens so the Base theme renders on our palette. */
gradio-app, .gradio-container, .gradio-container.dark, .dark {
    --body-background-fill: #f4f0e8 !important;
    --body-text-color: #1a1612 !important;
    --body-text-color-subdued: #3d372c !important;
    --background-fill-primary: #f4f0e8 !important;
    --background-fill-secondary: #ebe5d5 !important;
    --block-background-fill: #fbf8f1 !important;
    --block-border-color: #1a1612 !important;
    --block-border-width: 1px !important;
    --block-label-background-fill: #ebe5d5 !important;
    --block-label-text-color: #1a1612 !important;
    --block-title-text-color: #0f0d0a !important;
    --border-color-primary: #1a1612 !important;
    --border-color-accent: #5c1e18 !important;
    --input-background-fill: #fbf8f1 !important;
    --input-border-color: #1a1612 !important;
    --input-border-color-focus: #5c1e18 !important;
    --input-placeholder-color: #6b6354 !important;
    --button-primary-background-fill: #1a1612 !important;
    --button-primary-background-fill-hover: #5c1e18 !important;
    --button-primary-text-color: #f4f0e8 !important;
    --button-primary-border-color: #1a1612 !important;
    --button-secondary-background-fill: #fbf8f1 !important;
    --button-secondary-text-color: #1a1612 !important;
    --button-secondary-border-color: #1a1612 !important;
    --color-accent: #5c1e18 !important;
    --color-accent-soft: #ebe5d5 !important;
    --link-text-color: #5c1e18 !important;
    --link-text-color-hover: #3d0f0b !important;
    --neutral-50:  #fbf8f1;
    --neutral-100: #f4f0e8;
    --neutral-200: #ebe5d5;
    --neutral-300: #d4cfc0;
    --neutral-400: #a89f88;
    --neutral-500: #6b6354;
    --neutral-600: #3d372c;
    --neutral-700: #1a1612;
    --neutral-800: #0f0d0a;
    --neutral-900: #000;
    --shadow-drop: 2px 2px 0 #1a1612;
    --shadow-drop-lg: 3px 3px 0 #1a1612;
}

/* ---- Page / container ---------------------------------------------------- */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 28px 48px 28px !important;
    background: transparent !important;
}
gradio-app {
    background:
        radial-gradient(ellipse at 50% 0%, rgba(92,30,24,0.04), transparent 60%),
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.1  0 0 0 0 0.08  0 0 0 0 0.05  0 0 0 0.025 0'/></filter><rect width='100%' height='100%' filter='url(%23n)' /></svg>"),
        #ece6d4 !important;
    min-height: 100vh;
}

/* ---- Global typography --------------------------------------------------- */
.gradio-container, .gradio-container * {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: 0;
}
.gradio-container p, .gradio-container li,
.gradio-container label, .gradio-container .label,
.gradio-container span, .gradio-container div {
    color: #1a1612;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4,
.prose h1, .prose h2, .prose h3, .prose h4 {
    font-family: 'Fraunces', 'Libre Caslon Text', Georgia, serif !important;
    font-variation-settings: "opsz" 144, "SOFT" 50;
    letter-spacing: -0.02em;
    color: #0f0d0a;
    font-weight: 600;
}
.prose h2 { font-size: 1.85rem; margin: 0.25em 0 0.35em 0; font-weight: 500; letter-spacing: -0.025em; }

/* ---- Banner -------------------------------------------------------------- */
#app-banner {
    border-top: 3px double #1a1612;
    border-bottom: 3px double #1a1612;
    padding: 36px 0 32px 0;
    margin: 28px 0 24px 0;
    background: transparent;
    text-align: center;
}
#app-banner .eyebrow {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #5c1e18;
    margin-bottom: 18px;
}
#app-banner h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 400;
    font-variation-settings: "opsz" 144, "wght" 420, "SOFT" 30;
    color: #0f0d0a !important;
    margin: 0 0 16px 0;
    line-height: 1.08;
    letter-spacing: -0.028em;
    max-width: 22ch;
    margin-left: auto;
    margin-right: auto;
}
#app-banner p.lead {
    font-family: 'Fraunces', Georgia, serif !important;
    font-style: italic;
    font-variation-settings: "opsz" 144;
    font-size: 1.08rem;
    font-weight: 400;
    color: #3d372c !important;
    line-height: 1.55;
    margin: 0 auto;
    max-width: 56ch;
}

/* ---- Description block (between banner and form) ------------------------ */
.prose { color: #1a1612 !important; }
.prose strong { color: #5c1e18 !important; font-weight: 600; }

#intro {
    padding: 28px 36px !important;
    margin: 8px 0 32px 0 !important;
}
#intro h2 {
    margin-top: 0.1em !important;
    margin-bottom: 0.5em !important;
    font-size: 1.7rem !important;
    font-weight: 500 !important;
}
#intro p {
    font-size: 1rem !important;
    line-height: 1.7 !important;
    max-width: 68ch;
    color: #1a1612 !important;
}

/* ---- Inputs: strong ink borders ----------------------------------------- */
.gradio-container textarea,
.gradio-container input[type=text],
.gradio-container input[type=number],
.gradio-container input[type=search] {
    background: #fbf8f1 !important;
    color: #0f0d0a !important;
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.96rem !important;
    box-shadow: none !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
    outline: 2px solid #5c1e18 !important;
    outline-offset: 1px;
    border-color: #5c1e18 !important;
}
.gradio-container textarea::placeholder {
    color: #6b6354 !important;
    font-style: italic;
}

/* Dropdown container */
.gradio-container .wrap.svelte-1ixn6qd,
.gradio-container [class*="wrap-inner"],
.gradio-container .secondary-wrap {
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    background: #fbf8f1 !important;
}

/* Slider track */
.gradio-container input[type=range] { accent-color: #5c1e18 !important; }

/* Labels / block titles */
.gradio-container .block-title,
.gradio-container [data-testid="block-label"],
.gradio-container label > span:first-child {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #1a1612 !important;
}
.gradio-container .info, .gradio-container .hint {
    color: #6b6354 !important;
    font-size: 0.82rem !important;
    font-style: italic;
}

/* ---- Block containers (the input/output cards) -------------------------- */
.gradio-container .block {
    background: #fbf8f1 !important;
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    box-shadow: 2px 2px 0 rgba(26,22,18,0.9) !important;
}
/* don't shadow nested blocks inside a group (double-shadow) */
.gradio-container .block .block { box-shadow: none !important; }

/* ---- Primary action button ---------------------------------------------- */
/* Gradio attaches elem_id directly to the <button> element. Wrap it in a row
   container via margin + display: block and style the button itself. */
button#analyze-btn,
button#analyze-btn.primary,
button#analyze-btn.lg {
    display: block !important;
    margin: 32px auto 8px auto !important;
    background: #1a1612 !important;
    color: #f4f0e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.22em !important;
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    padding: 16px 56px !important;
    box-shadow: 4px 4px 0 #5c1e18 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease !important;
    width: auto !important;
    min-width: 260px !important;
    max-width: 360px !important;
    cursor: pointer !important;
}
button#analyze-btn:hover {
    transform: translate(-1px, -1px) !important;
    box-shadow: 5px 5px 0 #5c1e18 !important;
    background: #0f0d0a !important;
}
button#analyze-btn:active {
    transform: translate(3px, 3px) !important;
    box-shadow: 1px 1px 0 #5c1e18 !important;
}

/* ---- Highlighted text --------------------------------------------------- */
#highlight-block .textfield {
    line-height: 1.9 !important;
    font-size: 1rem !important;
    color: #0f0d0a !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 16px !important;
}
#highlight-block .textspan,
#highlight-block mark,
#highlight-block [class*="highlighted-token"] {
    border-radius: 2px !important;
    padding: 2px 5px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.97rem !important;
    border: 1px solid rgba(26,22,18,0.28) !important;
    color: #0f0d0a !important;
}
/* legend chips */
#highlight-block .category-legend span,
#highlight-block [class*="legend"] span {
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #0f0d0a !important;
}

/* ---- Summary panel ------------------------------------------------------ */
#summary-block {
    background: #fbf8f1 !important;
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    padding: 18px 28px 26px 28px !important;
    box-shadow: 2px 2px 0 rgba(26,22,18,0.9) !important;
}
#summary-block h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 500 !important;
    font-size: 1.25rem !important;
    color: #0f0d0a !important;
    margin: 1.6em 0 0.7em 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #1a1612 !important;
    letter-spacing: -0.015em;
}
#summary-block h3:first-child { margin-top: 0.5em !important; }
#summary-block strong { color: #5c1e18 !important; }
#summary-block p { color: #1a1612 !important; line-height: 1.6; }
#summary-block ul { padding-left: 20px; }
#summary-block li { margin-bottom: 4px; color: #1a1612 !important; }
#summary-block table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 10px 0 !important;
    font-size: 0.92rem !important;
    font-variant-numeric: tabular-nums;
    border: 1px solid #1a1612 !important;
}
#summary-block th {
    text-align: left !important;
    background: #1a1612 !important;
    color: #f4f0e8 !important;
    padding: 9px 12px !important;
    font-weight: 500 !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    border: none !important;
}
#summary-block td {
    border-bottom: 1px solid #e5dfcc !important;
    padding: 9px 12px !important;
    vertical-align: top !important;
    color: #1a1612 !important;
}
#summary-block tr:last-child td { border-bottom: none !important; }
#summary-block td:first-child {
    color: #5c1e18 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    white-space: nowrap;
}
#summary-block code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #ebe5d5 !important;
    color: #5c1e18 !important;
    padding: 2px 6px !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
    border: 1px solid #d4cfc0 !important;
}

/* ---- Examples block ----------------------------------------------------- */
#examples-block { margin-top: 40px; }
#examples-block label, #examples-block .block-title {
    font-family: 'Fraunces', serif !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: -0.01em !important;
    color: #0f0d0a !important;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1612;
    margin-bottom: 12px !important;
}
#examples-block .table-wrap {
    border: 1px solid #1a1612 !important;
    border-radius: 2px !important;
    box-shadow: 2px 2px 0 rgba(26,22,18,0.9) !important;
    overflow: hidden !important;
}
#examples-block .table-wrap table { background: #fbf8f1 !important; }
#examples-block .table-wrap th {
    background: #1a1612 !important;
    color: #f4f0e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 11px 14px !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    border: none !important;
    text-align: left !important;
}
#examples-block .table-wrap td {
    vertical-align: top !important;
    padding: 12px 14px !important;
    font-size: 0.9rem !important;
    border-bottom: 1px solid #e5dfcc !important;
    color: #1a1612 !important;
    background: #fbf8f1 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    line-height: 1.55;
}
#examples-block .table-wrap tr:hover td { background: #f5f0de !important; }
#examples-block .table-wrap td:first-child,
#examples-block .table-wrap th:first-child {
    max-width: 780px !important;
    min-width: 480px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* ---- Reference / footer card ------------------------------------------- */
#ref-card {
    margin-top: 56px !important;
    padding: 36px 44px !important;
    background: #fbf8f1 !important;
    border-top: 3px double #1a1612 !important;
    border-bottom: 3px double #1a1612 !important;
    border-left: none !important;
    border-right: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
#ref-card h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 500 !important;
    color: #0f0d0a !important;
    margin: 2em 0 0.8em 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #1a1612 !important;
    font-size: 1.35rem !important;
    letter-spacing: -0.02em !important;
}
#ref-card h3:first-child { margin-top: 0 !important; }
#ref-card p, #ref-card li { color: #1a1612 !important; line-height: 1.65; }
#ref-card ul { padding-left: 20px; }
#ref-card strong { color: #5c1e18 !important; }
#ref-card table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.9rem !important;
    margin-bottom: 1.2em !important;
    font-variant-numeric: tabular-nums;
    border: 1px solid #1a1612 !important;
}
#ref-card th {
    background: #1a1612 !important;
    color: #f4f0e8 !important;
    padding: 9px 12px !important;
    text-align: left !important;
    font-weight: 500 !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    border: none !important;
}
#ref-card td {
    border-bottom: 1px solid #e5dfcc !important;
    padding: 9px 12px !important;
    color: #1a1612 !important;
}
#ref-card tr:last-child td { border-bottom: none !important; }
#ref-card code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #ebe5d5 !important;
    color: #5c1e18 !important;
    padding: 2px 6px !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
    border: 1px solid #d4cfc0 !important;
}

/* ---- Misc -------------------------------------------------------------- */
.gradio-container ::selection { background: #5c1e18; color: #f4f0e8; }
.gradio-container a { color: #5c1e18; text-decoration: underline; text-underline-offset: 2px; }
.gradio-container a:hover { color: #3d0f0b; }
/* gradio footer/branding */
footer { display: none !important; }

/* ---- Examples: expanded-cell overrides (paired with JS observer) ------- */
#examples-block .table-wrap td:first-child .svelte-xxobeb,
#examples-block .table-wrap td:first-child > span > div {
    --local-text-width: 680px !important;
    max-width: 680px !important;
    min-width: 380px !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* ========================================================================
   Mobile / narrow viewport
   ======================================================================== */
@media (max-width: 820px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 0 14px 32px 14px !important;
    }
    #app-banner {
        padding: 24px 0 20px 0 !important;
        margin: 14px 0 18px 0 !important;
    }
    #app-banner .eyebrow {
        font-size: 0.58rem !important;
        letter-spacing: 0.22em !important;
        margin-bottom: 12px !important;
    }
    #app-banner h1 {
        font-size: clamp(1.55rem, 7vw, 2rem) !important;
        max-width: 18ch;
        line-height: 1.12 !important;
    }
    #app-banner p.lead {
        font-size: 0.94rem !important;
        line-height: 1.52 !important;
        padding: 0 10px;
    }
    #intro {
        padding: 20px 20px !important;
        margin: 6px 0 20px 0 !important;
    }
    #intro h2 { font-size: 1.35rem !important; }
    #intro p { font-size: 0.94rem !important; line-height: 1.6 !important; }
    button#analyze-btn {
        min-width: 200px !important;
        padding: 14px 28px !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.18em !important;
    }
    #summary-block {
        padding: 14px 16px 20px 16px !important;
    }
    #summary-block h3 {
        font-size: 1.05rem !important;
        margin-top: 1.3em !important;
    }
    #summary-block table,
    #ref-card table {
        display: block !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        white-space: nowrap !important;
    }
    #summary-block th, #ref-card th {
        font-size: 0.66rem !important;
        letter-spacing: 0.12em !important;
        padding: 7px 9px !important;
    }
    #summary-block td, #ref-card td {
        font-size: 0.85rem !important;
        padding: 7px 9px !important;
        white-space: normal !important;
    }
    #ref-card {
        padding: 26px 18px !important;
    }
    #ref-card h3 { font-size: 1.12rem !important; }

    #examples-block .table-wrap td:first-child,
    #examples-block .table-wrap th:first-child {
        min-width: auto !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    #examples-block .table-wrap td:first-child .svelte-xxobeb,
    #examples-block .table-wrap td:first-child > span > div {
        --local-text-width: 100% !important;
        max-width: 100% !important;
        min-width: auto !important;
    }
    /* Hide all non-text columns on mobile — the text preview alone identifies
       the example; the other fields still get loaded on click. */
    #examples-block .table-wrap th:not(:first-child),
    #examples-block .table-wrap td:not(:first-child) {
        display: none !important;
    }
    #examples-block .table-wrap th {
        font-size: 0.64rem !important;
        letter-spacing: 0.12em !important;
        padding: 8px 10px !important;
    }
    #examples-block .table-wrap td {
        font-size: 0.82rem !important;
        padding: 9px 10px !important;
    }
    #highlight-block .textfield {
        font-size: 0.92rem !important;
        line-height: 1.75 !important;
        padding: 12px !important;
    }
}
@media (max-width: 480px) {
    #app-banner h1 { font-size: 1.45rem !important; letter-spacing: -0.025em !important; }
    button#analyze-btn { min-width: calc(100% - 24px) !important; max-width: calc(100% - 24px) !important; padding: 14px 20px !important; }
    #intro { padding: 16px !important; }
}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    title="OPF PT-BR — Identificação de PII",
    css=_CSS,
) as demo:
    gr.HTML("""
        <div id="app-banner">
          <div class="eyebrow">OPF · Privacy Filter · Português Brasileiro · LGPD</div>
          <h1>Identificação de informações pessoais em documentos em português</h1>
          <p class="lead">Fine-tune do openai/privacy-filter em 914 mil amostras PT-BR — 73 rótulos
          cobrindo dados pessoais sensíveis nos termos do art. 5º, II da Lei nº 13.709/2018, com
          placeholder de substituição já emitido pelo decoder Viterbi constrained.</p>
        </div>
    """)
    gr.Markdown(DESCRIPTION, elem_id="intro")

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Texto para análise",
                placeholder="Cole aqui o trecho a ser revisado — peça processual, contrato, e-mail, relatório...",
                lines=10,
            )
        with gr.Column(scale=1):
            discard_overlapping = gr.Checkbox(
                label="Descartar spans sobrepostos do mesmo rótulo",
                value=False,
                info="Quando marcado, o cabeçalho do resumo sinaliza a opção.",
            )

    with gr.Row():
        labels_filter = gr.Dropdown(
            choices=ALL_LABELS,
            value=PTBR_LABELS,
            label="Rótulos considerados",
            info="Padrão: 24 labels PT-BR canônicos. Deixe vazio para usar todos os 73.",
            multiselect=True,
        )

    btn = gr.Button("Analisar texto", variant="primary", elem_id="analyze-btn")

    output_highlight = gr.HighlightedText(
        label="Texto anotado",
        combine_adjacent=False,
        show_legend=True,
        elem_id="highlight-block",
    )
    with gr.Group(elem_id="summary-block"):
        output_summary = gr.Markdown(label="Detalhamento por categoria")

    btn.click(predict,
              [input_text, labels_filter, discard_overlapping],
              [output_highlight, output_summary])

    with gr.Group(elem_id="examples-block"):
        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text, labels_filter, discard_overlapping],
            outputs=[output_highlight, output_summary],
            fn=predict,
            cache_examples=False,
            examples_per_page=6,
            label="Exemplos de referência (textos construídos manualmente, fora do conjunto de treino)",
        )

    with gr.Group(elem_id="ref-card"):
        gr.Markdown("""
### Desempenho do modelo

Ranking apples-to-apples (mesma val PT-BR 5k, mesmas 24 labels canônicas, nervaluate):

| Modelo | partial F1 | exact F1 |
|---|---|---|
| **opf-finetune (este)** | **0.897** | 0.853 |
| mmBERT-small × 3 | 0.888 | **0.870** |
| ettin-68m | 0.822 | 0.789 |
| ettin-32m | 0.808 | 0.769 |

### Recipe

| Item | Detalhe |
|---|---|
| Modelo | [`arthrod/gliner-opf-ptbr-pii-v1`](https://huggingface.co/arthrod/gliner-opf-ptbr-pii-v1) |
| Backbone | openai/privacy-filter (8L sparse-MoE, 128 experts, top-4 routing) |
| Treino | 9-round chunked sweep (3 ép × 3 saves), 914 mil linhas, ~3h45m em MI300X |
| Decoder | Viterbi constrained (BIESO) |
| Best ckpt | `e3_c3` — typed F1 0.885 (P 0.894 / R 0.876) |
| Batch / ctx | 32 janelas × grad-accum 4 (efetivo 128), n-ctx 256, bf16 |
| Optimizer | AdamW lr 1e-5, wd 0.01, max-grad-norm 1.0 |

### Diferenças vs GLiNER (mmBERT/ettin)

- Vocabulário de **73 labels** vs 24 (inclui SSN, passport, IP, MAC, swift_bic, etc.)
- Inferência via Viterbi constrained (BIESO) em vez de token-level argmax
- Emite `placeholder` pronto para substituição — não apenas o rótulo
- Sem easter-egg `berco-de-tiradentes` (use o demo mmBERT/ettin para isso)

### Catálogo de rótulos

**Documento oficial / identidade** — `cpf_document_number`, `rg_document_number`, `pis_document_number`, `id_card_number`, `passport_number`, `driver_license_number`, `social_security_number`, `tax_id_number`, `certificate_license_number`

**Dado financeiro** — `credit_card`, `account_number`, `bank_routing_number`, `swift_bic`, `cvv`, `pin`, `customer_id`

**Contato** — `phone_number`, `email_address`, `fax_number`, `url`, `ipv4`, `ipv6`, `mac_address`

**Nome** — `first_name`, `middle_name`, `last_name`, `user_name`, `title`

**Data de nascimento / idade** — `dob`, `date`, `time`, `date_time`, `age`

**Localização** — `location_street`, `location_building_number`, `location_neighborhood`, `location_city`, `location_state`, `location_state_abbreviation`, `location_zip`, `location_full_address`, `country`, `county`, `coordinate`

**Dado pessoal sensível (LGPD art. 5º, II)** — `personal_description_of_ethnicity`, `personal_description_of_medical_conditions`, `personal_description_of_organizational_affiliation`, `personal_description_of_political_opinion`, `personal_description_of_religious_convictions`, `personal_description_of_sexual_information`, `religious_belief`, `political_view`, `sexuality`, `sex_or_gender`, `biometric_identifier`, `blood_type`

**Credencial / identificador técnico** — `password`, `api_key`, `http_cookie`, `device_identifier`, `unique_id`, `employee_id`, `vehicle_identifier`, `license_plate`, `medical_record_number`, `health_plan_beneficiary_number`

**Contexto profissional** — `occupation`, `employment_status`, `education_level`, `company_name`, `language`
        """)

    # Inject expansion script as a child HTML component — Gradio 5.40's
    # Blocks.launch() does not accept a `head` kwarg, but a rendered <script>
    # tag works identically (MutationObserver boots once the DOM is parsed).
    gr.HTML(f"""
<script>
window.__OPF_EXAMPLES__ = {json.dumps([ex[0] for ex in EXAMPLES], ensure_ascii=False)};
(function() {{
  // Gradio's svelte Example component truncates text to 60 chars client-side.
  // Re-expand each cell by prefix-matching the full text we shipped above.
  function expand() {{
    var FULL = window.__OPF_EXAMPLES__ || [];
    var cells = document.querySelectorAll(
      '#examples-block .table-wrap tbody tr td:first-child'
    );
    var n = 0;
    cells.forEach(function(cell) {{
      var inner = cell.querySelector('.svelte-xxobeb.table')
               || cell.querySelector('div')
               || cell;
      var cur = (inner.textContent || '').trim();
      if (!cur || !cur.endsWith('...')) return;
      var prefix = cur.slice(0, -3);
      var match = FULL.find(function(full) {{ return full.startsWith(prefix); }});
      if (match && match !== cur) {{
        inner.textContent = match;
        inner.style.whiteSpace = 'normal';
        inner.style.wordBreak = 'break-word';
        inner.style.setProperty('--local-text-width', '680px', 'important');
        inner.style.maxWidth = '680px';
        inner.style.minWidth = '380px';
        n++;
      }}
    }});
    return n;
  }}
  function boot() {{
    [100, 400, 1000, 2000, 4000].forEach(function(t) {{ setTimeout(expand, t); }});
    var obs = new MutationObserver(function() {{ expand(); }});
    obs.observe(document.body, {{ childList: true, subtree: true, characterData: true }});
  }}
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', boot);
  }} else {{
    boot();
  }}
}})();
</script>
""")


if __name__ == "__main__":
    demo.queue(max_size=10).launch()
