"""Saying which generation of a corpus a result came from.

Both schema corpora are generated outside this repository and neither is
pinned by anything a fresh clone gets. A run recorded a hash per document and
nothing about the set they came from, so two generations could be compared
without anyone noticing.
"""

from __future__ import annotations

import pytest

from oold_llm_bench.corpus.provenance import corpus_digest


def _corpus(root, **files):
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_the_same_corpus_digests_the_same(tmp_path):
    a = _corpus(tmp_path / "a", **{}) if False else tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        _corpus(root, **{"X.json": '{"title": "X"}', "Y.json": '{"title": "Y"}'})
    assert corpus_digest(a) == corpus_digest(b)


def test_changed_content_changes_the_digest(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    _corpus(root, **{"X.json": '{"title": "X"}'})
    before = corpus_digest(root)
    (root / "X.json").write_text('{"title": "Z"}', encoding="utf-8")
    assert corpus_digest(root) != before


def test_a_rename_changes_the_digest(tmp_path):
    """The name is hashed beside the bytes, so a moved schema is a change."""
    root = tmp_path / "d"
    root.mkdir()
    _corpus(root, **{"X.json": '{"title": "X"}'})
    before = corpus_digest(root)
    (root / "X.json").rename(root / "W.json")
    assert corpus_digest(root) != before


def test_a_missing_file_is_visible_in_the_count(tmp_path):
    """A corpus that half arrived is the likelier accident, not one that moved."""
    root = tmp_path / "e"
    root.mkdir()
    _corpus(root, **{"X.json": "{}", "Y.json": "{}", "Z.json": "{}"})
    assert corpus_digest(root).files == 3
    (root / "Z.json").unlink()
    assert corpus_digest(root).files == 2


def test_an_empty_directory_raises_rather_than_digesting_nothing(tmp_path):
    root = tmp_path / "f"
    root.mkdir()
    with pytest.raises(ValueError, match="nothing to digest"):
        corpus_digest(root)


def test_it_describes_itself_for_a_record(tmp_path):
    root = tmp_path / "g"
    root.mkdir()
    _corpus(root, **{"X.json": "{}"})
    described = corpus_digest(root).describe()
    assert set(described) == {"digest", "files"}
    assert described["files"] == 1


class TestResolvingAModule:
    """A digest is the pin, so a path has to earn the same trust as a download."""

    def _module(self, root, **files):
        from oold_llm_bench.corpus.provenance import Module, corpus_digest

        root.mkdir()
        _corpus(root, **files)
        found = corpus_digest(root)
        return Module(name="test", repo_id="OO-LD/test", subdirectory="schemas", digest=str(found), files=found.files)

    def test_a_matching_directory_is_returned_unchanged(self, tmp_path):
        from oold_llm_bench.corpus.provenance import resolve_module

        root = tmp_path / "ok"
        module = self._module(root, **{"X.json": "{}"})
        assert resolve_module(module, root) == root

    def test_a_different_generation_is_refused(self, tmp_path):
        """The whole point. Two generations are not comparable, and a run
        against the wrong one produces numbers nobody can place."""
        from oold_llm_bench.corpus.provenance import resolve_module

        root = tmp_path / "moved"
        module = self._module(root, **{"X.json": '{"unit": "Ha"}'})
        (root / "X.json").write_text('{"unit": "ha"}', encoding="utf-8")
        with pytest.raises(ValueError, match="not comparable"):
            resolve_module(module, root)

    def test_the_message_names_the_digest_that_arrived(self, tmp_path):
        """The likeliest reason to see this is a module that moved on purpose,
        and the next step is to pin the new one."""
        from oold_llm_bench.corpus.provenance import corpus_digest, resolve_module

        root = tmp_path / "named"
        module = self._module(root, **{"X.json": "{}"})
        (root / "Y.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=str(corpus_digest(root))):
            resolve_module(module, root)

    def test_the_environment_variable_is_checked_too(self, tmp_path, monkeypatch):
        from oold_llm_bench.corpus.provenance import SCHEMAS_ENV, resolve_module

        root = tmp_path / "env"
        module = self._module(root, **{"X.json": "{}"})
        monkeypatch.setenv(SCHEMAS_ENV, str(root))
        assert resolve_module(module) == root
        (root / "X.json").write_text('{"a": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="not comparable"):
            resolve_module(module)


def test_the_two_published_modules_are_pinned():
    """A module with no digest is a module nobody can check they obtained."""
    from oold_llm_bench.corpus.provenance import QUANTITIES, SCHEMAORG

    for module in (QUANTITIES, SCHEMAORG):
        assert len(module.digest) == 64
        assert module.files > 900
        assert module.repo_id.startswith("OO-LD/")
