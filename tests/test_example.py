from oold_llm_bench import hello


def test_hello() -> None:
    assert "oold-llm-bench" in hello()
