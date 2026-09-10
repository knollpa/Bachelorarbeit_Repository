import json
from pathlib import Path
from transform import variant_generator as vg


def test_output_path_maps_to_existing_naming(tmp_path):
    p = vg.output_path("01", "statistics_addition", base=tmp_path)
    assert p.name == "01_StatisticsAddition.html"


def test_load_key_messages_returns_list(tmp_path):
    kmf = tmp_path / "key_messages.json"
    kmf.write_text(json.dumps({"01": {"kernaussagen": ["A", "B"]}}), encoding="utf-8")
    assert vg.load_key_messages("01", path=kmf) == ["A", "B"]


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _Resp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


class _FakeClient:
    def __init__(self, blocks=None, stop_reason="end_turn"):
        self.calls = []
        self._blocks = blocks if blocks is not None else [_Block("<p>TRANSFORMED</p>")]
        self._stop_reason = stop_reason

    class _Messages:
        def __init__(self, outer): self.outer = outer
        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            return _Resp(self.outer._blocks, self.outer._stop_reason)

    @property
    def messages(self): return _FakeClient._Messages(self)


# ---------------------------------------------------------------------------
# Test 1: basic generate_variant — model, thinking, no temperature
# ---------------------------------------------------------------------------

def test_generate_variant_writes_file_and_calls_model(tmp_path):
    (tmp_path / "01_Original.html").write_text("<p>BASE</p>", encoding="utf-8")
    client = _FakeClient()
    out = vg.generate_variant(client, "01", "fluency_optimization",
                              processed_dir=tmp_path, key_messages=None,
                              model="claude-opus-4-8")
    assert out.read_text(encoding="utf-8") == "<p>TRANSFORMED</p>"
    assert out.name == "01_FluencyOptimization.html"
    call = client.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["thinking"] == {"type": "adaptive"}
    assert "<p>BASE</p>" in call["messages"][0]["content"]
    assert call["system"]  # non-empty
    assert "temperature" not in call


# ---------------------------------------------------------------------------
# Test 2: thinking blocks are skipped; text block is used
# ---------------------------------------------------------------------------

def test_generate_variant_skips_thinking_blocks(tmp_path):
    (tmp_path / "01_Original.html").write_text("<p>BASE</p>", encoding="utf-8")
    blocks = [_Block("", type="thinking"), _Block("<p>OK</p>", type="text")]
    client = _FakeClient(blocks=blocks)
    out = vg.generate_variant(client, "01", "fluency_optimization",
                              processed_dir=tmp_path)
    assert "<p>OK</p>" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 3: max_tokens truncation raises RuntimeError
# ---------------------------------------------------------------------------

def test_generate_variant_raises_on_max_tokens(tmp_path):
    (tmp_path / "01_Original.html").write_text("<p>BASE</p>", encoding="utf-8")
    client = _FakeClient(stop_reason="max_tokens")
    import pytest
    with pytest.raises(RuntimeError, match="max_tokens"):
        vg.generate_variant(client, "01", "fluency_optimization",
                            processed_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 4: conclusion_first with key_messages injects them into prompt
# ---------------------------------------------------------------------------

def test_generate_variant_conclusion_first_with_key_messages(tmp_path):
    (tmp_path / "01_Original.html").write_text("<p>BASE</p>", encoding="utf-8")
    client = _FakeClient()
    out = vg.generate_variant(client, "01", "conclusion_first",
                              processed_dir=tmp_path,
                              key_messages=["A", "B"])
    assert out.name == "01_ConclusionFirst.html"
    prompt_content = client.calls[0]["messages"][0]["content"]
    assert "1. A" in prompt_content
    assert "2. B" in prompt_content


# ---------------------------------------------------------------------------
# Test 5: generate_all produces exactly 21 variants (3 articles × 7 methods)
# ---------------------------------------------------------------------------

def test_generate_all_produces_21_variants(tmp_path):
    # Seed processed dir with three baseline articles
    for aid in ("01", "02", "03"):
        (tmp_path / f"{aid}_Original.html").write_text(f"<p>ARTICLE {aid}</p>",
                                                       encoding="utf-8")

    # Seed key_messages JSON for all three articles
    km_data = {
        "01": {"kernaussagen": ["KM1-A", "KM1-B"]},
        "02": {"kernaussagen": ["KM2-A", "KM2-B"]},
        "03": {"kernaussagen": ["KM3-A", "KM3-B"]},
    }
    km_file = tmp_path / "key_messages.json"
    km_file.write_text(json.dumps(km_data), encoding="utf-8")

    client = _FakeClient()
    written = vg.generate_all(client,
                              processed_dir=tmp_path,
                              key_messages_path=km_file)
    assert len(written) == 21
    assert len(client.calls) == 21
