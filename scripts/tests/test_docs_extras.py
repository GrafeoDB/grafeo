"""Post-build extras: Markdown copies, llms.txt links, stubs for old paths."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from docs_extras import (  # noqa: E402
    StubCollisionError,
    copy_markdown,
    copy_old_markdown,
    page_url,
    write_stubs,
)


@pytest.fixture
def tmp_site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    return site


@pytest.fixture
def tmp_docs(tmp_path):
    docs = tmp_path / "docs"
    (docs / "model-and-query/gql").mkdir(parents=True)
    (docs / "includes").mkdir()
    (docs / "index.md").write_text("# Home\n")
    (docs / "model-and-query/gql/patterns.md").write_text("# Patterns\n")
    (docs / "includes/x.md").write_text("snippet\n")
    return docs


TABLE = {"user-guide/gql/patterns.md": "model-and-query/gql/patterns.md"}


def test_page_url():
    assert page_url("user-guide/gql/patterns.md") == "user-guide/gql/patterns/"
    assert page_url("index.md") == "" and page_url("api/index.md") == "api/"


def test_stub_points_to_new_page_and_keeps_fragment(tmp_site):
    write_stubs(TABLE, tmp_site)
    html = (tmp_site / "user-guide/gql/patterns/index.html").read_text()
    assert 'url=/model-and-query/gql/patterns/"' in html and "location.hash" in html


def test_stub_is_noindex(tmp_site):
    write_stubs(TABLE, tmp_site)
    html = (tmp_site / "user-guide/gql/patterns/index.html").read_text()
    assert 'name="robots" content="noindex"' in html and 'rel="canonical"' in html


def test_stub_never_overwrites_page(tmp_site):
    page = tmp_site / "user-guide/gql/patterns/index.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html>real page</html>")
    with pytest.raises(StubCollisionError):
        write_stubs(TABLE, tmp_site)
    assert page.read_text() == "<html>real page</html>"


def test_stub_rewrite_is_idempotent(tmp_site):
    write_stubs(TABLE, tmp_site)
    assert write_stubs(TABLE, tmp_site) == 1


def test_identity_entries_write_no_stub(tmp_site):
    assert write_stubs({"index.md": "index.md"}, tmp_site) == 0
    assert not list(tmp_site.iterdir())


def test_markdown_copies_old_and_new(tmp_site, tmp_docs):
    assert copy_markdown(tmp_docs, tmp_site) == 2
    assert copy_old_markdown(TABLE, tmp_docs, tmp_site) == 1
    assert (tmp_site / "md/model-and-query/gql/patterns.md").is_file()
    assert (tmp_site / "md/user-guide/gql/patterns.md").is_file()
    assert not (tmp_site / "md/includes").exists()


def test_llms_txt_links_resolve():
    text = (ROOT / "docs/llms.txt").read_text(encoding="utf-8")
    links = re.findall(r"https://grafeo\.dev(/[^\s)]*)", text)
    assert links
    for link in links:
        path = link.strip("/")
        assert (
            not path
            or (ROOT / "docs" / path / "index.md").is_file()
            or (ROOT / "docs" / f"{path}.md").is_file()
        ), link
    old = json.loads((ROOT / "docs/redirects.json").read_text(encoding="utf-8"))
    for link in links:
        assert (link.strip("/") + "/index.md") not in {
            k for k, v in old.items() if k != v
        }, link
