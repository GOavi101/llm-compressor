from collections.abc import Callable
from copy import copy

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    fake_quantize,
)
from compressed_tensors.quantization.utils import calculate_range

from llmcompressor.modifiers.gptq.helpers import GPTQ_PRECISION, column_scale_window

__all__ = ["admm_quantize_weight", "QParamFitter"]

ADMM_RHO_INIT = 0.1
ADMM_RHO_GROWTH = 1.1
ADMM_CONVERGENCE_INTERVAL = 10
LOCAL_SEARCH_PAIRS_PER_STEP = 64

# maps candidate weights [batch, rows, columns] to stacked (scale, zero_point)
QParamFitter = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def admm_quantize_weight(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    global_scale: torch.Tensor | None,
    quant_args: QuantizationArgs,
    percdamp: float = 0.01,
    num_iterations: int = 300,
    rho_init: float = ADMM_RHO_INIT,
    rho_growth: float = ADMM_RHO_GROWTH,
    local_search_rounds: int = 5,
    fit_qparams: QParamFitter | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a batch of weights with ADMM-Q (https://arxiv.org/abs/2605.11222).

    Solves the same Hessian-weighted layer-wise problem as GPTQ, but updates all
    weights jointly. Each iteration solves for continuous weights in closed form,
    projects them onto the quantization grid, and raises the penalty which pulls
    the two together. The Hessian is diagonally preconditioned to unit diagonal,
    and eigendecomposed once so each iteration is two matmuls.

    If ``fit_qparams`` is given, the grid is refit once from the continuous iterate
    and kept only if it projects the iterate more closely and lowers the final
    loss. Integer grids are then refined by a pair-swap local search which moves
    pairs of weights by one grid step whenever that lowers the loss.

    Column order does not affect the result, so activation ordering is ignored.

    :param weights: weights with shape [batch, rows, columns]
    :param hessians: normalized Hessians with shape [batch, columns, columns]
    :param scale: stacked observer scales
    :param zero_point: stacked observer zero points
    :param global_scale: optional stacked observer global scales
    :param quant_args: quantization arguments used to find quantization parameters
    :param percdamp: dampening factor on hessian diagonal
    :param num_iterations: maximum number of ADMM iterations
    :param rho_init: initial penalty, relative to the unit-diagonal Hessian
    :param rho_growth: factor the penalty grows by every iteration
    :param local_search_rounds: maximum number of local search rounds, 0 disables
    :param fit_qparams: optional observer used to refit the grid
    :return: quantized weights, per-batch losses, RTN fallback flags, and the
        scales and zero points of the grid the weights lie on
    """
    if weights.ndim != 3 or hessians.ndim != 3:
        raise ValueError("weights and hessians must have shape [batch, ...]")
    if weights.shape[0] != hessians.shape[0]:
        raise ValueError("weights and hessians must have matching batch sizes")

    batch_size, num_rows, num_columns = weights.shape
    strategy = quant_args.strategy
    final_dtype = weights.dtype
    device = weights.device
    W_hat = weights.to(device=device, dtype=GPTQ_PRECISION)
    H = hessians.to(device=device, dtype=GPTQ_PRECISION)
    scale = scale.to(device=device)
    zero_point = zero_point.to(device=device)
    if global_scale is not None:
        global_scale = global_scale.to(device=device)
    del weights, hessians

    g_idx = None
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=device, dtype=torch.int) // divisor

    elementwise_args = copy(quant_args)
    elementwise_args.strategy = QuantizationStrategy.CHANNEL

    def expand_grid(scale, zero_point):
        return column_scale_window(
            scale,
            zero_point,
            global_scale,
            g_idx,
            quant_args,
            num_rows=num_rows,
            i1=0,
            i2=num_columns,
        )

    def project(x, eff_scale, eff_zp):
        return fake_quantize(x, eff_scale, eff_zp, elementwise_args)

    eff_scale, eff_zp = expand_grid(scale, zero_point)

    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        diag.masked_fill_(dead, 1.0)
        W_hat.masked_fill_(dead.unsqueeze(1), 0)
    diag.add_((percdamp * diag.mean(dim=-1)).unsqueeze(-1))

    # scale the problem so the Hessian has unit diagonal
    sqrt_diag = diag.sqrt()
    H.div_(sqrt_diag.unsqueeze(-1)).div_(sqrt_diag.unsqueeze(-2))
    sqrt_diag = sqrt_diag.unsqueeze(1)
    W_hat.mul_(sqrt_diag)

    used_rtn_fallback = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eigvals, eigvecs = _batched_eigh(H, used_rtn_fallback)
    HW_hat = torch.bmm(W_hat, H)

    # only grids without a shared global scale can be refit per module
    can_refresh = fit_qparams is not None and global_scale is None
    refreshed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pre_refresh = None

    D = W_hat.clone()
    V = torch.zeros_like(W_hat)
    Q = project(W_hat / sqrt_diag, eff_scale, eff_zp)
    Q_checkpoint = Q.clone()
    rho = rho_init
    for iteration in range(num_iterations):
        W = torch.bmm(HW_hat + rho * D - V, eigvecs)
        W.div_((eigvals + rho).unsqueeze(1))
        W = torch.bmm(W, eigvecs.transpose(-1, -2))

        target = W + V / rho
        Q = project(target / sqrt_diag, eff_scale, eff_zp)
        D = Q * sqrt_diag
        V.add_(W - D, alpha=rho)

        if (iteration + 1) % ADMM_CONVERGENCE_INTERVAL == 0:
            converged = (Q == Q_checkpoint).flatten(1).all(dim=1)
            if can_refresh and not refreshed.all():
                new_scale, new_zp = fit_qparams(target / sqrt_diag)
                new_scale = new_scale.to(device=device, dtype=scale.dtype)
                new_zp = new_zp.to(device=device, dtype=zero_point.dtype)
                new_eff_scale, new_eff_zp = expand_grid(new_scale, new_zp)
                Q_new = project(target / sqrt_diag, new_eff_scale, new_eff_zp)
                D_new = Q_new * sqrt_diag
                accept = ~refreshed & (
                    (D_new - target).square().sum(dim=(1, 2))
                    <= (D - target).square().sum(dim=(1, 2))
                )
                if accept.any():
                    if pre_refresh is None:
                        pre_refresh = (Q.clone(), scale.clone(), zero_point.clone())
                    else:
                        _select(accept, pre_refresh, (Q, scale, zero_point))
                    scale = _where(accept, new_scale, scale)
                    zero_point = _where(accept, new_zp, zero_point)
                    eff_scale, eff_zp = expand_grid(scale, zero_point)
                    V.add_(_where(accept, D - D_new, torch.zeros_like(D)), alpha=rho)
                    Q = _where(accept, Q_new, Q)
                    D = Q * sqrt_diag
                    refreshed |= accept
                    converged &= ~accept
                del new_scale, new_zp, new_eff_scale, new_eff_zp, Q_new, D_new

            if converged.all():
                break
            Q_checkpoint.copy_(Q)
        rho *= rho_growth
    del W, V, HW_hat, Q_checkpoint, target, eigvals, eigvecs

    if used_rtn_fallback.any():
        Q[used_rtn_fallback] = project(W_hat / sqrt_diag, eff_scale, eff_zp)[
            used_rtn_fallback
        ]

    if local_search_rounds > 0 and quant_args.type == QuantizationType.INT:
        Q = _pair_swap(
            Q, W_hat, H, sqrt_diag, eff_scale, eff_zp, quant_args, local_search_rounds
        )
    losses = _losses(Q * sqrt_diag, W_hat, H)

    # keep the observer grid for modules whose refreshed solution is worse
    if pre_refresh is not None:
        Q_old, scale_old, zp_old = pre_refresh
        eff_scale_old, eff_zp_old = expand_grid(scale_old, zp_old)
        if local_search_rounds > 0 and quant_args.type == QuantizationType.INT:
            Q_old = _pair_swap(
                Q_old,
                W_hat,
                H,
                sqrt_diag,
                eff_scale_old,
                eff_zp_old,
                quant_args,
                local_search_rounds,
            )
        losses_old = _losses(Q_old * sqrt_diag, W_hat, H)
        revert = refreshed & (losses_old < losses)
        Q = _where(revert, Q_old, Q)
        scale = _where(revert, scale_old, scale)
        zero_point = _where(revert, zp_old, zero_point)
        losses = torch.where(revert, losses_old, losses)

    return Q.to(final_dtype), losses, used_rtn_fallback, scale, zero_point


def _where(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Select ``a`` for batch items where ``mask`` is set, else ``b``"""
    return torch.where(mask.reshape(-1, *([1] * (a.ndim - 1))), a, b)


def _select(
    mask: torch.Tensor,
    dst: tuple[torch.Tensor, ...],
    src: tuple[torch.Tensor, ...],
) -> None:
    """Copy batch items where ``mask`` is set from ``src`` into ``dst``"""
    for d, s in zip(dst, src):
        d[mask] = s[mask]


def _losses(D: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """0.5 * tr(E H E^T) for each scaled error E = D - W_hat"""
    error = D - W_hat
    return (torch.bmm(error, H) * error).sum(dim=(1, 2)) / 2


@torch.no_grad()
def _pair_swap(
    Q: torch.Tensor,
    W_hat: torch.Tensor,
    H: torch.Tensor,
    sqrt_diag: torch.Tensor,
    eff_scale: torch.Tensor,
    eff_zp: torch.Tensor | None,
    quant_args: QuantizationArgs,
    num_rounds: int,
) -> torch.Tensor:
    """
    Refine integer quantized weights by moving pairs of input columns of each row
    by at most one grid step. Each round randomly pairs up the columns, and every
    step each row applies its best move among a batch of pairs. The loss
    decomposes over rows, so moves on different rows do not interact, and every
    applied move strictly lowers the loss.
    """
    batch_size, num_rows, num_columns = Q.shape
    device = Q.device
    q_min, q_max = calculate_range(quant_args, device)
    zp = 0.0 if eff_zp is None else eff_zp
    codes = torch.round(Q / eff_scale + zp)
    step = eff_scale * sqrt_diag
    D = Q * sqrt_diag
    G = torch.bmm(W_hat - D, H)
    del D

    signs = torch.tensor(
        [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)],
        device=device,
        dtype=Q.dtype,
    )
    sign_a, sign_b = signs[:, 0], signs[:, 1]
    batch_index = torch.arange(batch_size, device=device).view(-1, 1)
    row_index = torch.arange(num_rows, device=device).view(1, -1)
    generator = torch.Generator().manual_seed(0)

    for _ in range(num_rounds):
        perm = torch.randperm(num_columns, generator=generator).to(device)
        num_pairs = num_columns // 2
        pairs = perm[: 2 * num_pairs].view(num_pairs, 2)
        improved = False
        for start in range(0, num_pairs, LOCAL_SEARCH_PAIRS_PER_STEP):
            col_a, col_b = pairs[start : start + LOCAL_SEARCH_PAIRS_PER_STEP].unbind(1)

            def moves(col, sign):
                # step size of each candidate move, zero where it leaves the grid
                c = codes[..., col].unsqueeze(-1)
                s = step[..., col].unsqueeze(-1) * sign
                return s * ((c + sign >= q_min) & (c + sign <= q_max))

            delta_a = moves(col_a, sign_a)  # [batch, rows, pairs, moves]
            delta_b = moves(col_b, sign_b)
            g_a = G[..., col_a].unsqueeze(-1)
            g_b = G[..., col_b].unsqueeze(-1)
            h_aa = H[:, col_a, col_a].view(batch_size, 1, -1, 1)
            h_bb = H[:, col_b, col_b].view(batch_size, 1, -1, 1)
            h_ab = H[:, col_a, col_b].view(batch_size, 1, -1, 1)
            delta_loss = (
                delta_a.square() * h_aa
                + delta_b.square() * h_bb
                + 2 * delta_a * delta_b * h_ab
                - 2 * delta_a * g_a
                - 2 * delta_b * g_b
            ).flatten(2)

            best_loss, best = delta_loss.min(dim=-1)
            apply = best_loss < 0
            if not apply.any():
                continue
            improved = True

            pair = best // signs.shape[0]
            a = col_a[pair]
            b = col_b[pair]
            d_a = torch.where(
                apply, delta_a.flatten(2).gather(-1, best.unsqueeze(-1))[..., 0], 0
            )
            d_b = torch.where(
                apply, delta_b.flatten(2).gather(-1, best.unsqueeze(-1))[..., 0], 0
            )
            codes[batch_index, row_index, a] += torch.sign(d_a)
            codes[batch_index, row_index, b] += torch.sign(d_b)
            G -= (
                d_a.unsqueeze(-1) * H[batch_index, a]
                + d_b.unsqueeze(-1) * H[batch_index, b]
            )
        if not improved:
            break

    return (codes - zp) * eff_scale


def _batched_eigh(
    hessians: torch.Tensor, used_rtn_fallback: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecompose each Hessian. Hessians which fail to decompose are replaced with
    the identity and marked for RTN fallback
    """
    try:
        return torch.linalg.eigh(hessians)
    except torch.linalg.LinAlgError:
        pass

    num_columns = hessians.shape[-1]
    eigvals = torch.ones(
        hessians.shape[:-1], dtype=hessians.dtype, device=hessians.device
    )
    eigvecs = torch.eye(
        num_columns, dtype=hessians.dtype, device=hessians.device
    ).repeat(hessians.shape[0], 1, 1)
    for index in range(hessians.shape[0]):
        try:
            eigvals[index], eigvecs[index] = torch.linalg.eigh(hessians[index])
        except torch.linalg.LinAlgError:
            used_rtn_fallback[index] = True
    return eigvals, eigvecs
