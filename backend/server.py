from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("BACKEND_PORT", "8000"))
MODEL_ID = os.environ.get("MODEL_ID", "sanim05/GPT2-disease_text_generation")
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
MODEL_SOURCE = MODEL_PATH or MODEL_ID
MODEL_LOCAL_ONLY = os.environ.get("MODEL_LOCAL_ONLY", "0") == "1"
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "1") == "1"

_MODEL_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None
_DEVICE = None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def get_model() -> tuple[Any, Any, str]:
    global _TOKENIZER, _MODEL, _DEVICE

    if _TOKENIZER is not None and _MODEL is not None and _DEVICE is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    with _MODEL_LOCK:
        if _TOKENIZER is None or _MODEL is None or _DEVICE is None:
            device = resolve_device()
            model_kwargs: dict[str, Any] = {}

            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_SOURCE,
                local_files_only=MODEL_LOCAL_ONLY,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_SOURCE,
                local_files_only=MODEL_LOCAL_ONLY,
                **model_kwargs,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.to(device)
            model.eval()

            _TOKENIZER = tokenizer
            _MODEL = model
            _DEVICE = device

    return _TOKENIZER, _MODEL, _DEVICE


def generate_text(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()

    if not prompt:
        raise ValueError("Trường 'prompt' không được để trống.")

    max_new_tokens = int(clamp(float(payload.get("max_new_tokens", 80)), 16, 512))
    temperature = clamp(float(payload.get("temperature", 0.8)), 0.2, 1.5)
    top_p = clamp(float(payload.get("top_p", 0.95)), 0.5, 1.0)
    repetition_penalty = clamp(float(payload.get("repetition_penalty", 1.1)), 1.0, 1.6)

    tokenizer, model, device = get_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    continuation = (
        generated_text[len(prompt) :].lstrip()
        if generated_text.startswith(prompt)
        else generated_text
    )

    return {
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "device": device,
        "prompt": prompt,
        "generated_text": generated_text,
        "continuation": continuation,
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "DiseaseGPTBackend/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._write_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_id": MODEL_ID,
                    "model_source": MODEL_SOURCE,
                    "device": _DEVICE,
                    "loaded": _MODEL is not None,
                },
            )
            return

        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "Không tìm thấy endpoint."},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/generate":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "Không tìm thấy endpoint."},
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            result = generate_text(payload)
            self._write_json(HTTPStatus.OK, result)
        except ValueError as error:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # noqa: BLE001
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Lỗi backend: {error}"},
            )


def run() -> None:
    if PRELOAD_MODEL:
        print(f"Preloading model from: {MODEL_SOURCE}")
        tokenizer, _, device = get_model()
        print(f"Loaded tokenizer vocab size: {len(tokenizer)}")
        print(f"Using device: {device}")

    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Backend running at http://{HOST}:{PORT}")
    print(f"Model: {MODEL_ID}")
    print(f"Source: {MODEL_SOURCE}")
    server.serve_forever()


if __name__ == "__main__":
    run()
