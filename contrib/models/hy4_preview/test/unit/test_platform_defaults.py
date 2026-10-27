"""Attention-path defaults per platform (``resolve_platform_defaults``)."""
import pytest

from hy4_stubs import model  # noqa: F401  (stubs vllm / vllm_neuron)

FLAGS = ("HY4_DEC_CP", "HY4_DEC_SPARSE_NKI", "HY4_PREFILL_SPARSE_NKI", "HY4_PREFILL_SPARSE_BACKEND")


@pytest.fixture(autouse=True)
def _restore_flags(monkeypatch):
    # resolve_platform_defaults writes module globals; monkeypatch restores them.
    for name in ("DEC_CP", "DEC_SPARSE_NKI", "PREFILL_SPARSE_NKI", "PREFILL_SPARSE_BACKEND"):
        monkeypatch.setattr(model, name, getattr(model, name))
    yield


@pytest.mark.parametrize(
    "target, expected",
    [
        ("trn2", (True, True, True, "v1")),
        ("trn3", (True, True, False, "cte")),
        ("", (False, False, False, "v1")),
    ],
)
def test_defaults(monkeypatch, target, expected):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.setattr(model, "_platform_target", lambda: target)
    model.resolve_platform_defaults()
    assert (
        model.DEC_CP, model.DEC_SPARSE_NKI, model.PREFILL_SPARSE_NKI, model.PREFILL_SPARSE_BACKEND
    ) == expected


def test_env_overrides(monkeypatch):
    monkeypatch.setattr(model, "_platform_target", lambda: "trn2")
    monkeypatch.setenv("HY4_PREFILL_SPARSE_NKI", "0")
    monkeypatch.setenv("HY4_DEC_SPARSE_NKI", "0")
    monkeypatch.delenv("HY4_DEC_CP", raising=False)
    model.resolve_platform_defaults()
    assert (model.DEC_CP, model.DEC_SPARSE_NKI, model.PREFILL_SPARSE_NKI) == (True, False, False)
    monkeypatch.setattr(model, "_platform_target", lambda: "trn3")
    monkeypatch.setenv("HY4_PREFILL_SPARSE_NKI", "1")
    monkeypatch.setenv("HY4_PREFILL_SPARSE_BACKEND", "v1")
    model.resolve_platform_defaults()
    assert model.PREFILL_SPARSE_NKI is True
    assert model.PREFILL_SPARSE_BACKEND == "v1"
