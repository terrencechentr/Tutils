import torch
from collections import defaultdict

def _format_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

@torch.no_grad()
def calculate_model_memory(model, include_buffers=True, include_grads=False, by_device=True):
    """
    Print parameter memory usage of a model. Units include bytes of tensors only.
    This does NOT include optimizer states or CUDA allocator overhead.
    """
    def tensor_bytes(t):
        return t.nelement() * t.element_size()

    per_dev = defaultdict(lambda: {"params":0, "grads":0, "buffers":0})
    trainable, frozen = 0, 0

    # parameters
    for p in model.parameters():
        dev = str(p.device)
        per_dev[dev]["params"] += tensor_bytes(p)
        (trainable if p.requires_grad else frozen)  # just to count
        if p.requires_grad:
            trainable += p.numel()
            if include_grads and p.grad is not None:
                per_dev[dev]["grads"] += tensor_bytes(p.grad)
        else:
            frozen += p.numel()

    # buffers (e.g., running stats)
    if include_buffers:
        for b in model.buffers():
            dev = str(b.device)
            per_dev[dev]["buffers"] += tensor_bytes(b)

    # totals
    total = {"params":0, "grads":0, "buffers":0}
    for dev, d in per_dev.items():
        for k in total: total[k] += d[k]

    # print
    print(f"Model: {model}")
    print("=== Model Memory Report (tensors only) ===")
    print(f"Trainable params: {trainable:,} | Frozen params: {frozen:,}")
    print("--------------------by device-------------------------")
    if by_device:
        for dev, d in per_dev.items():
            subtotal = d["params"] + d["grads"] + d["buffers"]
            print(f"- Device: {dev:>8} | total={_format_bytes(subtotal)} "
                  f"(params={_format_bytes(d['params'])}, "
                  f"grads={_format_bytes(d['grads'])}, "
                  f"buffers={_format_bytes(d['buffers'])})")
    print("--------------------total-------------------------")
    grand_total = total["params"] + total["grads"] + total["buffers"]
    print(f"Grand total (all devices): {_format_bytes(grand_total)} "
          f"[params={_format_bytes(total['params'])}"
          f"{', grads='+_format_bytes(total['grads']) if include_grads else ''}"
          f"{', buffers='+_format_bytes(total['buffers']) if include_buffers else ''}]")
    print("------------------------------------------------")
    return grand_total  # bytes
