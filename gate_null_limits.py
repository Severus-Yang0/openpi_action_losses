"""Check each loss in action_losses.py equals plain cross-entropy at its null setting.

Each loss is meant to be a strict generalisation of the baseline: at eps 0, T 0, lam 0 it
must return the baseline's value AND gradient to float precision, or a measured difference
between arms could be an implementation artefact. Runs on random logits, no model, no GPU.

    python gate_null_limits.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import action_losses as al


def main() -> int:
    cost, gather, lo, _hi, vb = al._table()
    rng = np.random.default_rng(0)
    b, ell = 3, 24
    full_vocab = int(np.asarray(gather).max()) + 1 + 4

    # A realistic sequence: structural prefix, a run of action tokens, terminator, padding.
    targets = np.full((b, ell), 7, dtype=np.int32)
    targets[:, 2:-4] = lo + rng.integers(0, vb, size=(b, ell - 6))
    loss_mask = np.zeros((b, ell), dtype=bool)
    loss_mask[:, 1:-2] = True
    logits = jnp.asarray(rng.normal(0, 2.0, size=(b, ell, full_vocab)).astype(np.float32))
    targets, loss_mask = jnp.asarray(targets), jnp.asarray(loss_mask)

    def run(kind, **knobs):
        old = {k: getattr(al, k) for k in ("KIND", *knobs)}
        al.KIND = kind
        for k, v in knobs.items():
            setattr(al, k, v)
        val = al.action_loss(logits, targets, loss_mask)
        grad = jax.grad(lambda x: jnp.mean(al.action_loss(x, targets, loss_mask)))(logits)
        for k, v in old.items():
            setattr(al, k, v)
        return val, grad

    ref, ref_grad = run("ce")
    ok = True
    print("null limits (must be 0, baseline reproduced exactly):")
    for name, kind, knobs in [
        ("ls   eps=0", "ls", {"LABEL_SMOOTH": 0.0}),
        ("soft T=1e-4", "soft", {"TEMPERATURE": 1e-4}),
        ("cost lam=0", "cost", {"LAM": 0.0}),
    ]:
        val, grad = run(kind, **knobs)
        dv = float(jnp.max(jnp.abs(val - ref)))
        dg = float(jnp.max(jnp.abs(grad - ref_grad)))
        good = max(dv, dg) < 1e-5
        ok &= good
        print(f"  {name:<14} |dloss| {dv:.3e}  |dgrad| {dg:.3e}  {'ok' if good else 'FAIL'}")

    print("\nknobs at their real settings (must differ from the baseline):")
    for name, kind, knobs in [
        ("ls   eps=0.1", "ls", {"LABEL_SMOOTH": 0.1}),
        ("soft T=0.05", "soft", {"TEMPERATURE": 0.05}),
        ("cost lam=1", "cost", {"LAM": 1.0}),
    ]:
        val, grad = run(kind, **knobs)
        dv = float(jnp.max(jnp.abs(val - ref)))
        dg = float(jnp.max(jnp.abs(grad - ref_grad)))
        good = dv > 1e-3 and dg > 1e-9
        ok &= good
        print(f"  {name:<14} |dloss| {dv:.3e}  |dgrad| {dg:.3e}  {'ok' if good else 'FAIL'}")

    # The cost table has to be readable for this tokenizer at all: if the character scheme
    # does not match, every pair lands in the different-length bucket, the table becomes
    # all-LARGE, and cost/soft silently become plain cross-entropy with a healthy loss.
    c = np.asarray(cost)
    credit = float((c < al.LARGE_COST).mean())
    neighbours = (c < al.LARGE_COST).sum(1)
    print(f"\ncost table {c.shape[0]}x{c.shape[0]}: {credit:.4f} of pairs have a real distance, "
          f"{float((neighbours > 1).mean()):.3f} of tokens have >1 neighbour "
          f"(median {np.median(neighbours[neighbours > 1]):.0f})")
    if credit < 0.01 or np.any(np.diag(c) != 0):
        print("FAIL: cost table is degenerate for this tokenizer")
        ok = False

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
