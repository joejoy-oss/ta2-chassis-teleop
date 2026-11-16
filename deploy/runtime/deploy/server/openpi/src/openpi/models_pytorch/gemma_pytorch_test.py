import pytest
import torch
from transformers.models.gemma import modeling_gemma

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch


def _rotary_embeddings(model: torch.nn.Module) -> list[modeling_gemma.GemmaRotaryEmbedding]:
    return [m for m in model.modules() if isinstance(m, modeling_gemma.GemmaRotaryEmbedding)]


def _assert_exact_inv_freq(model: torch.nn.Module) -> None:
    rotary = _rotary_embeddings(model)
    assert len(rotary) == 2  # the PaliGemma language model and the action expert
    for module in rotary:
        expected, _ = module.rope_init_fn(module.config, module.inv_freq.device)
        assert module.inv_freq.dtype == torch.float32
        assert torch.equal(module.inv_freq, expected.to(torch.float32))


@pytest.mark.parametrize("pi05", [False, True])
def test_bfloat16_conversion_keeps_rotary_frequencies_exact(pi05):
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=pi05, pytorch_compile_mode=None
    )
    # The constructor converts to bfloat16 (config.dtype); create_trained_policy converts once more.
    model = pi0_pytorch.PI0Pytorch(config)
    paligemma_with_expert = model.paligemma_with_expert
    assert paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    _assert_exact_inv_freq(model)

    paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    _assert_exact_inv_freq(model)

    # Casting the bf16 model up for a float32 comparison must not bring back bf16-rounded frequencies.
    _assert_exact_inv_freq(model.to(torch.float32))


@pytest.mark.parametrize("pi05", [False, True])
def test_float32_conversion_restores_rotary_frequencies_rounded_by_a_plain_cast(pi05):
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=pi05, pytorch_compile_mode=None
    )
    model = pi0_pytorch.PI0Pytorch(config)
    paligemma_with_expert = model.paligemma_with_expert
    # A plain .to(bfloat16) rounds the frequency buffers; the float32 branch must recompute them, not just cast up.
    paligemma_with_expert.to(dtype=torch.bfloat16)
    assert all(m.inv_freq.dtype == torch.bfloat16 for m in _rotary_embeddings(model))

    paligemma_with_expert.to_bfloat16_for_selected_params("float32")
    _assert_exact_inv_freq(model)
