from rbdpo.indicators.registry import DEFAULT_REGISTRY
from rbdpo.search.space import build_config_space, sample_and_validate


def test_space_samples_without_error():
    cs = build_config_space(3, 1, DEFAULT_REGISTRY, seed=42)
    samples = sample_and_validate(cs, n_samples=100)
    assert len(samples) == 100
    assert all(isinstance(s, dict) for s in samples)


def test_seed_determinism():
    cs1 = build_config_space(3, 1, DEFAULT_REGISTRY, seed=123)
    cs2 = build_config_space(3, 1, DEFAULT_REGISTRY, seed=123)
    seq1 = [cs1.sample_configuration() for _ in range(5)]
    seq2 = [cs2.sample_configuration() for _ in range(5)]
    assert seq1 == seq2
