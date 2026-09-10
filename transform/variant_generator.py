"""Reproduzierbare Erzeugung der 21 Artikelvarianten aus den 3 Baselines."""
from __future__ import annotations
import json
from pathlib import Path

from . import prompts

ARTICLE_IDS = ("01", "02", "03")
DEFAULT_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_KEY_MESSAGES = Path(__file__).resolve().parents[1] / "data" / "annotations" / "key_messages.json"
MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000


def output_path(article_id: str, method_key: str, base: Path = DEFAULT_PROCESSED) -> Path:
    return Path(base) / f"{article_id}_{prompts.METHODS[method_key].filename}.html"


def load_baseline(article_id: str, processed_dir: Path = DEFAULT_PROCESSED) -> str:
    return (Path(processed_dir) / f"{article_id}_Original.html").read_text(encoding="utf-8")


def load_key_messages(article_id: str, path: Path = DEFAULT_KEY_MESSAGES) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data[article_id]["kernaussagen"])


def generate_variant(client, article_id: str, method_key: str, *,
                     processed_dir: Path = DEFAULT_PROCESSED,
                     key_messages: list[str] | None = None,
                     model: str = MODEL) -> Path:
    article = load_baseline(article_id, processed_dir)
    system, user = prompts.build_prompt(method_key, article, key_messages=key_messages)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Output truncated (stop_reason=max_tokens) for {article_id}/{method_key}; "
            "increase MAX_TOKENS."
        )
    html = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not html:
        raise RuntimeError(
            f"No text block returned for {article_id}/{method_key} (stop_reason={resp.stop_reason})."
        )
    out = output_path(article_id, method_key, processed_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def generate_all(client, *, processed_dir: Path = DEFAULT_PROCESSED,
                 key_messages_path: Path = DEFAULT_KEY_MESSAGES,
                 model: str = MODEL) -> list[Path]:
    written: list[Path] = []
    for article_id in ARTICLE_IDS:
        for method_key, method in prompts.METHODS.items():
            kms = load_key_messages(article_id, key_messages_path) if method.needs_key_messages else None
            written.append(generate_variant(
                client, article_id, method_key,
                processed_dir=processed_dir, key_messages=kms, model=model))
    return written
