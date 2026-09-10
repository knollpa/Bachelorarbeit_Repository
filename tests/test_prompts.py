import pytest
from transform import prompts


def test_all_seven_methods_registered():
    assert set(prompts.METHODS) == {
        "statistics_addition", "quotation_addition", "fluency_optimization",
        "authoritative_tone", "logical_structure", "conclusion_first", "json_ld",
    }


def test_filename_suffix_matches_existing_data():
    assert prompts.METHODS["statistics_addition"].filename == "StatisticsAddition"
    assert prompts.METHODS["json_ld"].filename == "JSON-LD"
    assert prompts.METHODS["conclusion_first"].filename == "ConclusionFirst"


def test_build_prompt_injects_article():
    system, user = prompts.build_prompt("statistics_addition", "<p>HELLO</p>")
    assert "expert editor" in system
    assert "<p>HELLO</p>" in user
    assert "{article}" not in user


def test_conclusion_first_requires_and_injects_key_messages():
    kms = ["Fact one.", "Fact two."]
    system, user = prompts.build_prompt("conclusion_first", "<p>A</p>", key_messages=kms)
    assert "1. Fact one." in user
    assert "2. Fact two." in user
    assert "{key_messages}" not in user


def test_conclusion_first_without_key_messages_raises():
    with pytest.raises(ValueError):
        prompts.build_prompt("conclusion_first", "<p>A</p>")


def test_non_conclusion_method_ignores_key_messages():
    _, user = prompts.build_prompt("fluency_optimization", "<p>A</p>", key_messages=["x"])
    assert "key messages" not in user.lower()


def test_format_key_messages_numbers_entries():
    assert prompts.format_key_messages(["a", "b"]) == "1. a\n2. b"
