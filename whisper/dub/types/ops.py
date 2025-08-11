from functools import lru_cache
import numpy as np
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; run `pip install --pre triton`")

@triton.jit
def dtw_kernel(cost_ptr, trace_ptr, x_ptr, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    for k in range(1, N + M + 1):
        tl.debug_barrier()
        p0 = cost_ptr + (k - 1) * cost_stride
        p1 = cost_ptr + k * cost_stride
        p2 = cost_ptr + k * cost_stride + 1
        c0 = tl.load(p0 + offsets, mask=mask, other=float('inf'))
        c1 = tl.load(p1 + offsets, mask=mask, other=float('inf'))
        c2 = tl.load(p2 + offsets, mask=mask, other=float('inf'))
        x_row = tl.load(x_ptr + (k - 1) * x_stride + offsets, mask=mask, other=0.0)
        cost_vals = x_row + tl.minimum(tl.minimum(c0, c1), c2)
        dst_cost_ptr = cost_ptr + (k + 1) * cost_stride + 1
        tl.store(dst_cost_ptr + offsets, cost_vals, mask=mask)
        dst_trace_ptr = trace_ptr + (k + 1) * trace_stride + 1
        eq_c2 = (c2 <= c0) & (c2 <= c1)
        eq_c1 = (c1 <= c0) & (c1 <= c2)
        eq_c0 = (c0 <= c1) & (c0 <= c2)
        tl.store(dst_trace_ptr + offsets, 2, mask=mask & eq_c2)
        tl.store(dst_trace_ptr + offsets, 1, mask=mask & eq_c1)
        tl.store(dst_trace_ptr + offsets, 0, mask=mask & eq_c0)

@lru_cache(maxsize=None)
def median_kernel(filter_width: int):
    assert filter_width % 2 == 1 and filter_width > 0
    @triton.jit
    def _kernel(y_ptr, x_ptr, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < y_stride
        row_ptr = x_ptr + pid * x_stride
        rows = [tl.load(row_ptr + offsets + i, mask=mask, other=0.0) for i in range(filter_width)]
        for i in range(filter_width // 2 + 1):
            for j in range(filter_width - i - 1):
                smaller = tl.where(rows[j] < rows[j + 1], rows[j], rows[j + 1])
                larger = tl.where(rows[j] > rows[j + 1], rows[j], rows[j + 1])
                rows[j], rows[j + 1] = smaller, larger
        median_row = rows[filter_width // 2]
        out_ptr = y_ptr + pid * y_stride
        tl.store(out_ptr + offsets, median_row, mask=mask)
    return _kernel

def median_filter_cuda(x: torch.Tensor, filter_width: int):
    if not torch.is_tensor(x):
        raise TypeError("Input must be torch.Tensor")
    if filter_width <= 0 or filter_width % 2 == 0:
        raise ValueError("filter_width must be positive odd integer")
    slices = x.contiguous().unfold(-1, filter_width, 1)
    if slices.numel() == 0:
        return x
    grid = int(np.prod(slices.shape[:-2]))
    kernel = median_kernel(filter_width)
    BLOCK_SIZE = min(256, 1 << (x.stride(-2) - 1).bit_length())
    y = torch.empty_like(slices[..., 0])
    kernel[(grid,)](y, x, x.stride(-2), y.stride(-2), BLOCK_SIZE=BLOCK_SIZE)
    return y
