import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, fake_quantize
from pydantic import ValidationError

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.admm_quantize import admm_quantize_weight
from llmcompressor.modifiers.gptq.base import _observer_fitter
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from tests.llmcompressor.modifiers.gptq.test_gptq_quantize import (
    _make_observed_linear,
    _quantization_inputs,
)


def _correlated_hessian(in_features, seed=0, num_tokens=2048):
    """Hessian of correlated inputs with a few outlier channels, as seen in LLMs"""
    gen = torch.Generator().manual_seed(seed)
    mix = torch.randn(in_features, in_features, generator=gen) / in_features**0.5
    inputs = torch.randn(num_tokens, in_features, generator=gen) @ (
        mix + torch.eye(in_features)
    )
    inputs[:, :4] *= 20
    return 2 * inputs.T @ inputs / num_tokens


def _hessian_error(quantized, weight, hessian):
    error = quantized - weight
    return (error @ hessian * error).sum().item()


def _solve(solver, inputs, quant_args, **kwargs):
    weights, hessians, scales, zero_points, global_scales = inputs
    return solver(
        weights=weights.clone(),
        hessians=hessians.clone(),
        scale=scales,
        zero_point=zero_points,
        global_scale=global_scales,
        quant_args=quant_args,
        **kwargs,
    )


@pytest.mark.parametrize(
    "quant_args",
    [
        QuantizationArgs(num_bits=4, symmetric=True, strategy="channel"),
        QuantizationArgs(num_bits=4, symmetric=False, strategy="channel"),
        QuantizationArgs(num_bits=4, symmetric=True, strategy="group", group_size=16),
        QuantizationArgs(num_bits=8, symmetric=True, strategy="tensor"),
        QuantizationArgs(
            num_bits=8, symmetric=True, strategy="block", block_structure=[8, 16]
        ),
    ],
)
@torch.no_grad()
def test_admm_weights_lie_on_quantization_grid(quant_args):
    module = _make_observed_linear(64, 32, quant_args)
    inputs = _quantization_inputs([module], [_correlated_hessian(64)])
    _, _, scales, zero_points, _ = inputs

    quantized, losses, used_rtn_fallback, *_ = _solve(
        admm_quantize_weight, inputs, quant_args
    )

    requantized = fake_quantize(quantized[0], scales[0], zero_points[0], quant_args)
    assert torch.equal(requantized, quantized[0])
    assert quantized.shape == (1, *module.weight.shape)
    assert losses.shape == (1,) and losses[0] >= 0
    assert not used_rtn_fallback.any()


@pytest.mark.parametrize("num_bits", [2, 3])
@torch.no_grad()
def test_admm_beats_gptq_at_low_bits(num_bits):
    quant_args = QuantizationArgs(
        num_bits=num_bits, symmetric=True, strategy="group", group_size=32
    )
    module = _make_observed_linear(256, 128, quant_args)
    hessian = _correlated_hessian(256)
    inputs = _quantization_inputs([module], [hessian])
    _, _, scales, zero_points, _ = inputs
    rtn = fake_quantize(module.weight, scales[0], zero_points[0], quant_args)

    gptq, _, _ = _solve(quantize_weight, inputs, quant_args)
    admm, *_ = _solve(admm_quantize_weight, inputs, quant_args)

    gptq_error = _hessian_error(gptq[0], module.weight, hessian)
    admm_error = _hessian_error(admm[0], module.weight, hessian)
    assert admm_error < gptq_error < _hessian_error(rtn, module.weight, hessian)


@torch.no_grad()
def test_admm_zeroes_dead_columns():
    quant_args = QuantizationArgs(num_bits=4, symmetric=True, strategy="channel")
    module = _make_observed_linear(32, 16, quant_args)
    hessian = _correlated_hessian(32)
    hessian[3, :] = 0
    hessian[:, 3] = 0

    inputs = _quantization_inputs([module], [hessian])
    quantized, losses, *_ = _solve(admm_quantize_weight, inputs, quant_args)

    assert torch.isfinite(quantized).all() and torch.isfinite(losses).all()
    assert (quantized[0, :, 3] == 0).all()


@torch.no_grad()
def test_admm_batched_matches_single():
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=16
    )
    modules = [_make_observed_linear(64, 32, quant_args, seed=seed) for seed in (0, 1)]
    hessians = [_correlated_hessian(64, seed=seed) for seed in (0, 1)]

    inputs = _quantization_inputs(modules, hessians)
    batched, *_ = _solve(admm_quantize_weight, inputs, quant_args)

    single = [
        _solve(
            admm_quantize_weight,
            [None if x is None else x[index : index + 1] for x in inputs],
            quant_args,
        )[0][0]
        for index in range(len(modules))
    ]

    for index, expected in enumerate(single):
        assert torch.allclose(batched[index], expected, atol=1e-5)


def test_gptq_modifier_solver_validation():
    assert GPTQModifier(scheme="W4A16").solver == "gptq"
    assert GPTQModifier(scheme="W4A16", solver="admm").solver == "admm"
    with pytest.raises(ValidationError):
        GPTQModifier(scheme="W4A16", solver="sgd")
    with pytest.raises(ValidationError):
        GPTQModifier(scheme="W4A16", solver="admm", admm_iterations=0)
    with pytest.raises(ValidationError):
        GPTQModifier(scheme="W4A16", solver="admm", admm_local_search_rounds=-1)


@torch.no_grad()
def test_gptq_modifier_compresses_with_admm():
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=16
    )
    module = _make_observed_linear(64, 48, quant_args)
    module.weight_scale = torch.nn.Parameter(torch.empty(48, 4), requires_grad=False)
    module.weight_zero_point = torch.nn.Parameter(
        torch.empty(48, 4, dtype=quant_args.zp_dtype), requires_grad=False
    )
    expected_module = _make_observed_linear(64, 48, quant_args)
    inputs = _quantization_inputs([expected_module], [_correlated_hessian(64)])
    expected, *_ = _solve(admm_quantize_weight, inputs, quant_args)

    modifier = GPTQModifier(solver="admm")
    modifier._module_names[module] = "model.layers.0.mlp.down_proj"
    modifier._hessians[module] = _correlated_hessian(64)
    modifier._num_samples[module] = torch.tensor(1.0)
    modifier.compress_modules()

    assert modifier._num_compressed_modules == 1
    assert torch.allclose(module.weight, expected[0], atol=1e-5)


@torch.no_grad()
def test_admm_nvfp4_weights_lie_on_quantization_grid():
    quant_args = QuantizationArgs(
        num_bits=4,
        type="float",
        symmetric=True,
        strategy="tensor_group",
        group_size=16,
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    )
    module = _make_observed_linear(64, 32, quant_args)
    inputs = _quantization_inputs([module], [_correlated_hessian(64)])
    _, _, scales, zero_points, global_scales = inputs

    quantized, losses, _, out_scales, _ = _solve(
        admm_quantize_weight, inputs, quant_args
    )

    requantized = fake_quantize(
        quantized[0], scales[0], zero_points[0], quant_args, global_scales[0]
    )
    assert torch.equal(requantized, quantized[0])
    assert torch.equal(out_scales, scales)
    assert torch.isfinite(losses).all()


@pytest.mark.parametrize("symmetric", [True, False])
@torch.no_grad()
def test_admm_local_search_lowers_loss_on_grid(symmetric):
    quant_args = QuantizationArgs(
        num_bits=3, symmetric=symmetric, strategy="group", group_size=32
    )
    module = _make_observed_linear(256, 128, quant_args)
    hessian = _correlated_hessian(256)
    inputs = _quantization_inputs([module], [hessian])
    _, _, scales, zero_points, _ = inputs

    plain, plain_loss, *_ = _solve(
        admm_quantize_weight, inputs, quant_args, local_search_rounds=0
    )
    searched, searched_loss, *_ = _solve(admm_quantize_weight, inputs, quant_args)

    requantized = fake_quantize(searched[0], scales[0], zero_points[0], quant_args)
    assert torch.equal(requantized, searched[0])
    assert searched_loss < plain_loss
    assert _hessian_error(searched[0], module.weight, hessian) < _hessian_error(
        plain[0], module.weight, hessian
    )


@torch.no_grad()
def test_admm_grid_refresh_returns_grid_of_weights():
    quant_args = QuantizationArgs(
        num_bits=3,
        symmetric=False,
        strategy="group",
        group_size=32,
        observer="memoryless_mse",
    )
    module = _make_observed_linear(128, 64, quant_args)
    inputs = _quantization_inputs([module], [_correlated_hessian(128)])

    quantized, losses, _, scales, zero_points = _solve(
        admm_quantize_weight,
        inputs,
        quant_args,
        fit_qparams=_observer_fitter([module]),
    )

    requantized = fake_quantize(quantized[0], scales[0], zero_points[0], quant_args)
    assert torch.equal(requantized, quantized[0])
    assert torch.isfinite(losses).all()


@torch.no_grad()
def test_admm_grid_refresh_with_unchanged_grid_is_noop():
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=16
    )
    module = _make_observed_linear(64, 32, quant_args)
    inputs = _quantization_inputs([module], [_correlated_hessian(64)])
    _, _, scales, zero_points, _ = inputs

    expected, *_ = _solve(admm_quantize_weight, inputs, quant_args)
    refreshed, *_ = _solve(
        admm_quantize_weight,
        inputs,
        quant_args,
        fit_qparams=lambda candidates: (scales, zero_points),
    )

    assert torch.equal(refreshed, expected)
