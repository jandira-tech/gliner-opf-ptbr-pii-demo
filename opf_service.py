"""Taxonomy, example catalogue and span post-processing for the OPF PT-BR demo.

This module is deliberately **free of any ML dependency** (no torch, no ``opf``):
it only knows about labels, categories, placeholders and how to turn the spans
produced by the OPF runtime into the payload the frontend renders.  That keeps
the interesting logic unit-testable on a machine without a GPU.

The label taxonomy below is the one shipped with the model card
(``arthrod/gliner-opf-ptbr-pii-v1``): 24 canonical PT-BR labels + 47 extra
cross-source labels = 71 PII labels.  The checkpoint's *label space* holds 73
classes, because it also contains the ``O`` (background) class and the
``berco_de_tiradentes`` easter egg; neither is a PII label a user can filter on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

REPO_ID = "arthrod/gliner-opf-ptbr-pii-v1"
MODEL_ID = REPO_ID

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------

#: The canonical PT-BR set used for the apples-to-apples comparison against
#: the GLiNER baselines (identical to the original Gradio app).
PTBR_LABELS: tuple[str, ...] = (
    "cpf_document_number",
    "rg_document_number",
    "pis_document_number",
    "credit_card",
    "phone_number",
    "email_address",
    "first_name",
    "middle_name",
    "last_name",
    "dob",
    "location_street",
    "location_building_number",
    "location_neighborhood",
    "location_city",
    "location_state",
    "location_state_abbreviation",
    "location_zip",
    "location_full_address",
    "personal_description_of_ethnicity",
    "personal_description_of_medical_conditions",
    "personal_description_of_organizational_affiliation",
    "personal_description_of_political_opinion",
    "personal_description_of_religious_convictions",
    "personal_description_of_sexual_information",
)

#: The 47 extra labels (passport, IP, SWIFT, plate, credentials, ...).
EXTRA_LABELS: tuple[str, ...] = (
    "date",
    "company_name",
    "time",
    "id_card_number",
    "user_name",
    "url",
    "title",
    "occupation",
    "country",
    "age",
    "account_number",
    "social_security_number",
    "sex_or_gender",
    "tax_id_number",
    "password",
    "driver_license_number",
    "customer_id",
    "passport_number",
    "date_time",
    "employee_id",
    "education_level",
    "biometric_identifier",
    "vehicle_identifier",
    "coordinate",
    "religious_belief",
    "employment_status",
    "ipv4",
    "political_view",
    "certificate_license_number",
    "medical_record_number",
    "fax_number",
    "language",
    "county",
    "license_plate",
    "health_plan_beneficiary_number",
    "bank_routing_number",
    "pin",
    "mac_address",
    "swift_bic",
    "api_key",
    "http_cookie",
    "blood_type",
    "device_identifier",
    "ipv6",
    "cvv",
    "sexuality",
    "unique_id",
)

ALL_LABELS: tuple[str, ...] = PTBR_LABELS + EXTRA_LABELS

#: Category key -> (display name in PT-BR, labels).
#:
#: The category *keys* are the ones the design system uses for its CSS
#: modifiers (``.cath.sens``, ``.ent.cat-cred`` ...), the display names are the
#: ones written in the design mock.
CATEGORIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "doc": (
        "Documento oficial",
        (
            "cpf_document_number",
            "rg_document_number",
            "pis_document_number",
            "id_card_number",
            "passport_number",
            "driver_license_number",
            "social_security_number",
            "tax_id_number",
            "certificate_license_number",
        ),
    ),
    "fin": (
        "Dado financeiro",
        (
            "credit_card",
            "account_number",
            "bank_routing_number",
            "swift_bic",
            "cvv",
            "pin",
            "customer_id",
        ),
    ),
    "contact": (
        "Contato",
        (
            "phone_number",
            "email_address",
            "fax_number",
            "url",
            "ipv4",
            "ipv6",
            "mac_address",
        ),
    ),
    "name": (
        "Nome",
        ("first_name", "middle_name", "last_name", "user_name", "title"),
    ),
    "dob": (
        "Data de nascimento / idade",
        ("dob", "date", "time", "date_time", "age"),
    ),
    "loc": (
        "Localização",
        (
            "location_street",
            "location_building_number",
            "location_neighborhood",
            "location_city",
            "location_state",
            "location_state_abbreviation",
            "location_zip",
            "location_full_address",
            "country",
            "county",
            "coordinate",
        ),
    ),
    "sens": (
        "Dado pessoal sensível · LGPD art. 5º, II",
        (
            "personal_description_of_ethnicity",
            "personal_description_of_medical_conditions",
            "personal_description_of_organizational_affiliation",
            "personal_description_of_political_opinion",
            "personal_description_of_religious_convictions",
            "personal_description_of_sexual_information",
            "religious_belief",
            "political_view",
            "sexuality",
            "sex_or_gender",
            "biometric_identifier",
            "blood_type",
        ),
    ),
    "cred": (
        "Credencial / identificador técnico",
        (
            "password",
            "api_key",
            "http_cookie",
            "device_identifier",
            "unique_id",
            "employee_id",
            "vehicle_identifier",
            "license_plate",
            "medical_record_number",
            "health_plan_beneficiary_number",
        ),
    ),
    "prof": (
        "Contexto profissional",
        ("occupation", "employment_status", "education_level", "company_name", "language"),
    ),
}

#: Labels the model can emit that are not part of the documented taxonomy
#: (e.g. the ``berco_de_tiradentes`` easter egg) land here.
CUSTOM_CATEGORY = "custom"
CUSTOM_CATEGORY_LABEL = "Label customizado (zero-shot)"

#: Rendering order of the category blocks (design system order).
CATEGORY_ORDER: tuple[str, ...] = tuple(CATEGORIES.keys()) + (CUSTOM_CATEGORY,)

_LABEL_TO_CATEGORY: dict[str, str] = {
    label: key for key, (_name, labels) in CATEGORIES.items() for label in labels
}

#: Placeholder base names (OPF style) — the design's ``PH`` map, extended with
#: the extra labels so a placeholder stays short and readable.
PLACEHOLDER_BASES: dict[str, str] = {
    "cpf_document_number": "CPF",
    "rg_document_number": "RG",
    "pis_document_number": "PIS",
    "credit_card": "CARTAO",
    "phone_number": "TELEFONE",
    "email_address": "EMAIL",
    "first_name": "NOME",
    "middle_name": "NOME_MEIO",
    "last_name": "SOBRENOME",
    "dob": "DATA_NASC",
    "location_street": "LOGRADOURO",
    "location_building_number": "NUMERO",
    "location_neighborhood": "BAIRRO",
    "location_city": "CIDADE",
    "location_state": "UF",
    "location_state_abbreviation": "UF",
    "location_zip": "CEP",
    "location_full_address": "ENDERECO",
    "personal_description_of_medical_conditions": "SAUDE",
    "personal_description_of_religious_convictions": "RELIGIAO",
    "personal_description_of_political_opinion": "POLITICA",
    "personal_description_of_sexual_information": "SEXUALIDADE",
    "personal_description_of_organizational_affiliation": "AFILIACAO",
    "personal_description_of_ethnicity": "ETNIA",
    "age": "IDADE",
    "occupation": "PROFISSAO",
    "company_name": "EMPRESA",
    "account_number": "CONTA",
    "license_plate": "PLACA",
    "certificate_license_number": "REGISTRO",
    "education_level": "FORMACAO",
    "url": "URL",
    "title": "TITULO",
    # extra labels
    "id_card_number": "IDENTIDADE",
    "passport_number": "PASSAPORTE",
    "driver_license_number": "CNH",
    "social_security_number": "NISS",
    "tax_id_number": "CNPJ",
    "customer_id": "CLIENTE",
    "bank_routing_number": "BANCO",
    "swift_bic": "SWIFT",
    "cvv": "CVV",
    "pin": "PIN",
    "fax_number": "FAX",
    "ipv4": "IPV4",
    "ipv6": "IPV6",
    "mac_address": "MAC",
    "user_name": "USUARIO",
    "date": "DATA",
    "time": "HORA",
    "date_time": "DATA_HORA",
    "country": "PAIS",
    "county": "CONDADO",
    "coordinate": "COORDENADA",
    "sex_or_gender": "GENERO",
    "sexuality": "SEXUALIDADE",
    "religious_belief": "RELIGIAO",
    "political_view": "POLITICA",
    "blood_type": "SANGUE",
    "biometric_identifier": "BIOMETRIA",
    "password": "SENHA",
    "api_key": "API_KEY",
    "http_cookie": "COOKIE",
    "device_identifier": "DISPOSITIVO",
    "unique_id": "ID_UNICO",
    "employee_id": "MATRICULA",
    "vehicle_identifier": "VEICULO",
    "medical_record_number": "PRONTUARIO",
    "health_plan_beneficiary_number": "CARTEIRA_SAUDE",
    "employment_status": "SITUACAO",
    "language": "IDIOMA",
}


def category_of(label: str) -> str:
    """Return the category key for ``label`` (``custom`` when unknown)."""
    return _LABEL_TO_CATEGORY.get(label, CUSTOM_CATEGORY)


def category_label(key: str) -> str:
    """Return the PT-BR display name of a category key."""
    if key == CUSTOM_CATEGORY:
        return CUSTOM_CATEGORY_LABEL
    return CATEGORIES[key][0]


def placeholder_base(label: str) -> str:
    """Return the placeholder base name used for ``label``."""
    return PLACEHOLDER_BASES.get(
        label, label.upper().replace("_", " ").replace(" ", "_")
    )


def all_labels() -> list[str]:
    return list(ALL_LABELS)


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSpan:
    """A span as emitted by the OPF runtime (character offsets)."""

    label: str
    start: int
    end: int
    text: str
    #: The placeholder the runtime itself produced, e.g. ``<CPF_DOCUMENT_NUMBER>``.
    model_placeholder: str = ""

    @classmethod
    def from_runtime(cls, span: Any) -> "RawSpan":
        """Build a :class:`RawSpan` from an ``opf`` ``DetectedSpan``."""
        return cls(
            label=str(getattr(span, "label", "")),
            start=int(getattr(span, "start", 0)),
            end=int(getattr(span, "end", 0)),
            text=str(getattr(span, "text", "")),
            model_placeholder=str(getattr(span, "placeholder", "") or ""),
        )


def filter_spans(spans: Sequence[RawSpan], labels: Iterable[str] | None) -> list[RawSpan]:
    """Keep only the spans whose label is selected.

    ``labels=None`` means "no filter" (every span).  An *empty* ``labels``
    collection means "nothing selected" and therefore returns nothing — the
    frontend always sends an explicit selection.
    """
    if labels is None:
        return list(spans)
    keep = set(labels)
    return [s for s in spans if s.label in keep]


def discard_same_label_overlaps(spans: Sequence[RawSpan]) -> list[RawSpan]:
    """Drop spans that overlap another span carrying the *same* label.

    Mirrors ``opf``'s ``discard_overlapping_predicted_spans``: for every label
    the longest span wins and shorter/overlapping siblings are removed.
    """
    kept: list[RawSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        clash = False
        for other in kept:
            if other.label != span.label:
                continue
            if span.start < other.end and other.start < span.end:
                clash = True
                break
        if not clash:
            kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end))


def resolve_overlaps(spans: Sequence[RawSpan]) -> list[RawSpan]:
    """Return the non-overlapping subset used for painting and redaction.

    Spans are consumed left to right; a span that starts before the cursor has
    already been consumed is skipped.  This mirrors what the original Gradio
    app did to build its ``HighlightedText`` segments.
    """
    resolved: list[RawSpan] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if span.start < cursor:
            continue
        resolved.append(span)
        cursor = span.end
    return resolved


def assign_placeholders(spans: Sequence[RawSpan]) -> list[dict[str, Any]]:
    """Attach a stable, numbered placeholder to every span.

    The numbering is per placeholder base name (``[CPF_1]``, ``[CPF_2]``,
    ``[NOME_1]`` ...) so the same kind of datum stays traceable inside the
    redacted document without exposing it.
    """
    counters: dict[str, int] = {}
    payload: list[dict[str, Any]] = []
    for span in spans:
        base = placeholder_base(span.label)
        counters[base] = counters.get(base, 0) + 1
        payload.append(
            {
                "label": span.label,
                "category": category_of(span.label),
                "category_label": category_label(category_of(span.label)),
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "placeholder": f"[{base}_{counters[base]}]",
                "model_placeholder": span.model_placeholder,
            }
        )
    return payload


def redact(text: str, spans: Sequence[dict[str, Any]]) -> str:
    """Replace every span with its placeholder, right to left."""
    redacted = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        placeholder = span.get("placeholder") or ""
        if not placeholder:
            continue
        redacted = redacted[: span["start"]] + placeholder + redacted[span["end"] :]
    return redacted



def redact_overlapping(text: str, spans: Sequence[RawSpan]) -> str:
    """Cover the union of detected spans, including overlapping tails."""
    groups: list[list[RawSpan]] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if groups and span.start < max(item.end for item in groups[-1]):
            groups[-1].append(span)
        else:
            groups.append([span])
    edits = []
    counters: dict[str, int] = {}
    for group in groups:
        if len(group) == 1:
            base = placeholder_base(group[0].label)
            counters[base] = counters.get(base, 0) + 1
            placeholder = f"[{base}_{counters[base]}]"
        else:
            placeholder = "[DADO_PESSOAL]"
        edits.append({"start": min(item.start for item in group),
                      "end": max(item.end for item in group), "placeholder": placeholder})
    return redact(text, edits)

def analyze(
    text: str,
    raw_spans: Sequence[RawSpan],
    labels: Iterable[str] | None,
    discard_overlapping: bool = False,
) -> dict[str, Any]:
    """Turn raw runtime spans into the payload consumed by the frontend.

    Steps: filter by label -> optionally drop same-label overlaps -> resolve
    cross-label overlaps (the ones that cannot be painted) -> number the
    placeholders -> build the redacted text.
    """
    selected = filter_spans(raw_spans, labels)
    redaction_spans = list(selected)
    if discard_overlapping:
        selected = discard_same_label_overlaps(selected)
    resolved = resolve_overlaps(selected)
    payload_spans = assign_placeholders(resolved)
    return {
        "text": text,
        "spans": payload_spans,
        "all_spans": assign_placeholders(selected),
        "redacted_text": redact_overlapping(text, redaction_spans),
        "count": len(payload_spans),
        "hidden": len(selected) - len(resolved),
        "labels_seen": sorted({s.label for s in selected}),
    }


# --------------------------------------------------------------------------
# Catalogue served to the frontend
# --------------------------------------------------------------------------


def catalog(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the ``/api/meta`` payload (labels, categories, examples)."""
    return {
        "model": MODEL_ID,
        "repo": REPO_ID,
        "canonical": list(PTBR_LABELS),
        "labels": list(ALL_LABELS),
        "categories": [
            {
                "key": key,
                "label": CATEGORIES[key][0],
                "labels": list(CATEGORIES[key][1]),
            }
            for key in CATEGORIES
        ],
        "custom_category": {
            "key": CUSTOM_CATEGORY,
            "label": CUSTOM_CATEGORY_LABEL,
            "labels": [],
        },
        "placeholder_bases": dict(PLACEHOLDER_BASES),
        "examples": list(examples),
    }
