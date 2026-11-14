"""The docs tone checker: what it flags in prose and what it leaves alone."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_docs_tone import findings, main  # noqa: E402


def rules(text):
    return [r for _, r, _ in findings(text)]


def test_flags_person_in_prose_not_in_code():
    assert [
        r for _, r, _ in findings("You can run it.\n\n```python\nyou = 1\n```\n")
    ] == ["person"]


def test_flags_em_dash_and_buzzword():
    rs = {r for _, r, _ in findings("Fast — and robust.")}
    assert rs == {"em-dash", "buzzword"}


def test_pip_install_but_not_uv_pip():
    assert [r for _, r, _ in findings("```bash\npip install grafeo\n```")] == ["pip"]
    assert findings("```bash\nuv pip install grafeo\n```") == []


def test_tone_crlf():
    assert [r for _, r, _ in findings("We ship.\r\n")] == ["person"]


def test_ignores_indented_fences_inline_code_urls_comments_front_matter():
    text = (
        "---\ntitle: You and we\n---\n"
        '=== "Tab"\n\n    ```python\n    your = 1\n    ```\n\n'
        "Use `your` value. See https://example.com/you/we and [docs](../our/page.md).\n"
        "<!-- docs-check: skip we cannot run this -->\n"
        "<!--\nyou\n-->\n"
    )
    assert findings(text) == []


def test_line_numbers_are_page_lines():
    assert findings("---\na: b\n---\nfine\nWe do.\n") == [(5, "person", "We do.")]


def test_pronoun_i_but_not_roman_numeral_or_io():
    assert rules("I think so.") == ["person"]
    assert rules("I will do it.") == ["person"]
    assert rules("Part I covers storage. Disk I/O is fast.") == []


def test_word_boundaries():
    assert rules("us-east-1 and the US and a user, youth, weekly") == []
    assert rules("We are here, you will see.") == ["person"]
    assert rules("We're here, you'll see.") == ["person"]


def test_buzzwords_case_insensitive():
    assert rules("A Seamless, cutting-edge store.") == ["buzzword"]
    assert rules("Robustness is fine, a robust store is not.") == ["buzzword"]


def test_pip_in_inline_code_and_prose():
    assert rules("Run `pip install grafeo`.") == ["pip"]
    assert rules("Run `uv pip install grafeo`.") == []


def test_main_exit_codes_and_section_page(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_bytes(b"We ship.\n")
    (tmp_path / "ok.md").write_bytes(b"Clean.\n")
    assert main([str(tmp_path), "--section", "ok.md"]) == 0
    assert main([str(tmp_path), "--section", "a"]) == 1
    assert "a/x.md:1: person: We ship." in capsys.readouterr().out
    assert main([str(tmp_path)]) == 1


def test_em_dash_entities():
    for ent in ("&mdash;", "&MDASH;", "&#8212;", "&#x2014;", "&#X2014;", "&#08212;"):
        assert rules(f"Fast {ent} and small.") == ["em-dash"], ent
    assert rules("Write `&mdash;` in HTML.") == []


def test_buzzword_inflections_and_spaced_compounds():
    for w in (
        "utilizes",
        "utilized",
        "utilizing",
        "leverages",
        "leveraged",
        "facilitates",
        "facilitated",
        "holistically",
        "comprehensively",
        "blazingly",
        "seamlessly",
        "cutting edge",
        "next generation",
        "enterprise grade",
        "Cutting-Edge",
        "paradigms",
    ):
        assert rules(f"It is {w} here.") == ["buzzword"], w
    assert rules("Robustness and powerfully are left alone.") == []
