"""Checks for the taxonomy, the span post-processing and the HTTP API.

The model itself is not needed: the API tests inject a fake runtime that
returns the same span shape as ``opf``'s ``DetectedSpan``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opf_service  # noqa: E402
from opf_examples import EXAMPLES  # noqa: E402

CANONICAL = set(opf_service.PTBR_LABELS)


# --------------------------------------------------------------- taxonomy


def test_taxonomy_covers_every_label_exactly_once():
    seen = [label for _name, labels in opf_service.CATEGORIES.values() for label in labels]
    assert len(seen) == len(set(seen)), "a label is listed in two categories"
    assert set(seen) == set(opf_service.ALL_LABELS)
    assert len(opf_service.ALL_LABELS) == 71


def test_canonical_set_is_the_model_card_set():
    assert len(opf_service.PTBR_LABELS) == 24
    assert CANONICAL <= set(opf_service.ALL_LABELS)


def test_every_label_has_a_category_and_a_placeholder_base():
    for label in opf_service.ALL_LABELS:
        assert opf_service.category_of(label) != opf_service.CUSTOM_CATEGORY
        assert opf_service.placeholder_base(label)


def test_unknown_label_falls_back_to_custom():
    assert opf_service.category_of("berco_de_tiradentes") == opf_service.CUSTOM_CATEGORY


def test_placeholder_base_matches_the_design_map():
    assert opf_service.placeholder_base("cpf_document_number") == "CPF"
    assert opf_service.placeholder_base("first_name") == "NOME"
    # state and state abbreviation deliberately share the "UF" counter
    assert opf_service.placeholder_base("location_state") == "UF"
    assert opf_service.placeholder_base("location_state_abbreviation") == "UF"


def test_examples_preserve_both_sources():
    ids = [e["id"] for e in EXAMPLES]
    assert len(ids) == len(set(ids))
    origins = {e["origin"] for e in EXAMPLES}
    assert origins == {"design-demo", "gradio-app"}
    # the design deep links keep working
    for legacy in ("whats", "peticao", "prontuario", "cv", "ticket", "daiana"):
        assert legacy in ids


# ------------------------------------------------------------ post-process


def _span(label, start, end, text=None):
    return opf_service.RawSpan(
        label=label,
        start=start,
        end=end,
        text=text if text is not None else "x" * (end - start),
        model_placeholder=f"<{label.upper()}>",
    )


def test_filter_spans_by_label():
    spans = [_span("cpf_document_number", 0, 3), _span("first_name", 5, 9)]
    assert len(opf_service.filter_spans(spans, ["cpf_document_number"])) == 1
    assert opf_service.filter_spans(spans, []) == []
    assert len(opf_service.filter_spans(spans, None)) == 2


def test_discard_same_label_overlaps_keeps_the_longest():
    spans = [
        _span("first_name", 0, 10),
        _span("first_name", 2, 6),
        _span("cpf_document_number", 3, 7),
    ]
    kept = opf_service.discard_same_label_overlaps(spans)
    assert [(s.label, s.start, s.end) for s in kept] == [
        ("first_name", 0, 10),
        ("cpf_document_number", 3, 7),
    ]


def test_resolve_overlaps_drops_cross_label_overlap():
    spans = [_span("first_name", 0, 10), _span("location_city", 4, 12)]
    resolved = opf_service.resolve_overlaps(spans)
    assert [s.label for s in resolved] == ["first_name"]


def test_placeholders_are_numbered_per_base():
    spans = [
        _span("cpf_document_number", 0, 3),
        _span("first_name", 5, 9),
        _span("cpf_document_number", 12, 15),
    ]
    payload = opf_service.assign_placeholders(opf_service.resolve_overlaps(spans))
    assert [p["placeholder"] for p in payload] == ["[CPF_1]", "[NOME_1]", "[CPF_2]"]
    assert payload[0]["model_placeholder"] == "<CPF_DOCUMENT_NUMBER>"
    assert payload[0]["category"] == "doc"


def test_redact_replaces_spans_with_placeholders():
    text = "CPF 123 e nome Ana"
    spans = [_span("cpf_document_number", 4, 7), _span("first_name", 15, 18)]
    payload = opf_service.assign_placeholders(opf_service.resolve_overlaps(spans))
    assert opf_service.redact(text, payload) == "CPF [CPF_1] e nome [NOME_1]"


def test_analyze_end_to_end():
    text = "Maria, CPF 123.456.789-00"
    spans = [_span("first_name", 0, 5), _span("cpf_document_number", 11, 25)]
    out = opf_service.analyze(text, spans, ["first_name", "cpf_document_number"], False)
    assert out["count"] == 2
    assert out["hidden"] == 0
    assert out["redacted_text"] == "[NOME_1], CPF [CPF_1]"


# ------------------------------------------------------------------- API


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module

    class FakeRuntime:
        def redact(self, text):
            class Span:
                def __init__(self, label, start, end, text):
                    self.label = label
                    self.start = start
                    self.end = end
                    self.text = text
                    self.placeholder = f"<{label.upper()}>"

            return type(
                "R",
                (),
                {
                    "detected_spans": (
                        Span("first_name", 0, 5, "Maria"),
                        Span("cpf_document_number", 11, 25, "123.456.789-00"),
                    )
                },
            )()

    monkeypatch.setattr(app_module, "_runtime", FakeRuntime())
    with TestClient(app_module.create_app()) as test_client:
        yield test_client


def test_index_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "OPF PT-BR" in resp.text


def test_meta_exposes_the_taxonomy(client):
    data = client.get("/api/meta").json()
    assert len(data["canonical"]) == 24
    assert len(data["labels"]) == 71
    assert {c["key"] for c in data["categories"]} == {
        "doc", "fin", "contact", "name", "dob", "loc", "sens", "cred", "prof",
    }
    assert len(data["examples"]) == len(EXAMPLES)


def test_analyze_returns_real_spans(client):
    import app as app_module
    data = app_module.analyze_queued("Maria, CPF 123.456.789-00", ["first_name", "cpf_document_number"], False)
    assert data["count"] == 2
    assert data["redacted_text"] == "[NOME_1], CPF [CPF_1]"


def test_analyze_with_empty_label_selection_searches_all_labels(client):
    import app as app_module
    data = app_module.analyze_queued("Maria, CPF 123", [], False)
    assert data["count"] == 2


def test_analyze_rejects_empty_text(client):
    import app as app_module
    with pytest.raises(Exception):
        app_module.analyze_queued("   ", [], False)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_overlapping_entities_remain_visible_and_redaction_covers_union():
    text = 'abcdefghij'
    spans = [_span('first_name', 0, 5, 'abcde'), _span('last_name', 3, 10, 'defghij')]
    result = opf_service.analyze(text, spans, None, False)
    assert len(result['all_spans']) == 2
    assert result['redacted_text'] == '[DADO_PESSOAL]'
