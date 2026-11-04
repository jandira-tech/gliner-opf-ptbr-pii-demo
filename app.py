"""OPF PT-BR PII — Space entrypoint.

Serves the design-system frontend (the ``static/`` folder) and exposes the
**real** OPF inference behind Gradio’s queued analyze endpoint.

The Space keeps the ``gradio`` SDK: that is what makes the Space eligible for
ZeroGPU scheduling, and ``spaces.GPU`` runs the decorated function in a forked
worker that inherits the runtime loaded in the parent process.  Gradio is only
used as the ASGI host — the routes below are registered *before* Gradio's own
``/`` route, so the design page is what the Space serves.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# `spaces` patches the CUDA allocator, so it has to be imported before torch.
try:  # pragma: no cover - provided by the Space platform
    import spaces  # type: ignore
except ImportError:  # pragma: no cover - local development without the package
    spaces = None

import gradio as gr
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.routing import Mount, Route

import opf_service
from opf_examples import EXAMPLES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPO_ID = opf_service.REPO_ID
MODEL_ID = opf_service.MODEL_ID

_runtime: Any | None = None
_runtime_error: str | None = None
_runtime_device: str | None = None


def load_runtime() -> Any | None:
    """Download the checkpoint and build the OPF runtime.

    Called at import time on purpose: on ZeroGPU the decorated function runs
    in a *forked* worker, so the runtime has to be resident in the parent
    process before the first request.
    """
    global _runtime, _runtime_error, _runtime_device
    if _runtime is not None:
        return _runtime
    try:
        import torch
        from huggingface_hub import snapshot_download
        from opf import OPF

        device = os.getenv("OPF_DEVICE") or "cuda"
        path = snapshot_download(
            REPO_ID,
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "USAGE.txt",
                "finetune_summary.json",
            ],
        )
        _runtime = OPF(
            model=path,
            device=device,
            output_mode="typed",
            output_text_only=False,
        )
        _runtime_device = device
        _runtime_error = None
        print(f"[opf] runtime ready · device={device} · {path}", flush=True)
    except Exception as exc:  # pragma: no cover - depends on the environment
        _runtime = None
        _runtime_error = f"{type(exc).__name__}: {exc}"
        print(f"[opf] runtime unavailable · {_runtime_error}", file=sys.stderr, flush=True)
    return _runtime


if spaces is not None:  # pragma: no cover - depends on the platform
    _gpu = spaces.GPU
else:  # pragma: no cover - local development without the spaces package

    def _gpu(fn):  # type: ignore[misc]
        return fn


def infer_spans(text: str) -> list[opf_service.RawSpan]:
    """Run the OPF runtime over ``text`` inside the scheduled callback."""
    runtime = _runtime
    if runtime is None:
        raise RuntimeError(_runtime_error or "OPF runtime indisponível")
    result = runtime.redact(text)
    return [opf_service.RawSpan.from_runtime(span) for span in result.detected_spans]


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------

router = APIRouter()


class AnalyzeRequest(BaseModel):
    text: str = Field(default="", description="Texto a ser analisado.")
    labels: list[str] | None = Field(
        default=None,
        description=(
            "Rótulos selecionados. `null` = sem filtro (todos os rótulos); "
            "`[]` = nada selecionado."
        ),
    )
    discard_overlapping: bool = Field(
        default=False,
        description="Descarta spans sobrepostos que compartilham o mesmo rótulo.",
    )


@router.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "status": "ok" if _runtime is not None else "degraded",
        "model": MODEL_ID,
        "device": _runtime_device,
        "error": _runtime_error,
    }


@router.get("/api/meta")
def api_meta() -> dict[str, Any]:
    """Label taxonomy, category tree and the example catalogue."""
    return opf_service.catalog(EXAMPLES)


def api_analyze(payload: AnalyzeRequest):
    text = payload.text or ""
    if not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "empty_text", "detail": "Cole um texto para analisar."},
        )
    if _runtime is None and os.getenv("OPF_EAGER_LOAD", "1") != "0":
        load_runtime()
    started = time.perf_counter()
    try:
        spans = infer_spans(text)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "inference_failed",
                "detail": f"{type(exc).__name__}: {exc}",
            },
        )
    result = opf_service.analyze(
        text,
        spans,
        payload.labels or None,
        payload.discard_overlapping,
    )
    result.update(
        {
            "model": MODEL_ID,
            "device": _runtime_device,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    return result


@_gpu
def analyze_queued(text: str, labels: list[str] | None, discard_overlapping: bool) -> dict:
    result = api_analyze(AnalyzeRequest(text=text, labels=labels, discard_overlapping=discard_overlapping))
    if isinstance(result, JSONResponse):
        raise gr.Error("Não foi possível analisar o texto. Verifique o modelo e tente novamente.")
    return result


def create_app() -> gr.Server:
    server = gr.Server(title="OPF PT-BR PII")
    server.include_router(router)
    server.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    server.api(analyze_queued, name="analyze")

    @server.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    return server


if __name__ == "__main__":
    if os.getenv("OPF_EAGER_LOAD", "1") != "0":
        load_runtime()
    create_app().launch()
