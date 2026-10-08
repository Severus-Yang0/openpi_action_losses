"""Action-aware training losses for pi0-FAST. Drop-in; see README.md.

Copy this file to openpi/src/openpi/models/action_losses.py and set KIND below.

Cross-entropy penalises every wrong action token equally. A FAST action token is a run of
quantised DCT coefficients, and the DCT is orthonormal, so coefficient distance equals
action distance: swapping bin k for k' costs ((k - k') / scale)^2 of squared action error.
COST[b, b'] accumulates that over the coefficients a BPE token covers. Token pairs covering
different numbers of coefficients get no credit, because such a swap shifts everything
downstream and no local distance describes it.

Every loss here reduces to plain cross-entropy at its null setting (eps 0, T 0, lam 0),
exactly, including gradients. gate_null_limits.py checks that.
"""

from __future__ import annotations

import collections
import functools
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
KIND = "soft"  # "ce" | "ls" | "soft" | "cost"   (aug is separate, see NoisyActionEncoder)
LABEL_SMOOTH = 0.1  # ls
TEMPERATURE = 0.05  # soft
LAM = 1.0  # cost

# Must be the tokenizer training actually uses, or the distances are computed for the wrong
# vocabulary and cost/soft go quietly wrong. This is openpi's default. Only change it if your
# config sets Pi0FASTConfig.fast_model_tokenizer_kwargs["fast_tokenizer_path"] to something
# else; then put that value here and re-run gate_null_limits.py.
TOKENIZER = "physical-intelligence/fast"
CACHE_DIR = ".cache"  # where the cost table is kept between runs

PALIGEMMA_VOCAB_SIZE = 257152
FAST_SKIP_TOKENS = 128
LARGE_COST = 1.0e6  # exp(-LARGE/T) underflows to 0, so no soft credit


# ----------------------------------------------------------------------------------------
# The cost table
# ----------------------------------------------------------------------------------------
def _expansions(bpe, coeff_vocab: int) -> list[np.ndarray | None]:
    """Coefficient indices each BPE token stands for, or None if it is a special token.

    The tokenizer writes coefficient bin k as chr(k - min_token), so decoding a token and
    taking ord() of each character recovers the run.
    """
    out = []
    for b in range(bpe.get_vocab_size()):
        ks = np.array([ord(c) for c in bpe.decode([b])], dtype=np.int64)
        out.append(ks if ks.size and int(ks.max()) < coeff_vocab else None)
    return out


def build_cost_table(tokenizer: str = TOKENIZER, cache_dir: str | None = CACHE_DIR):
    """Return (COST (Vb, Vb) float32, GATHER (Vb,) int32 PaliGemma logit columns).

    Takes a minute or two to build, so it is cached on disk.
    """
    import huggingface_hub
    import tokenizers

    local = pathlib.Path(tokenizer)
    if local.is_dir():
        cfg_path = local / "processor_config.json"
        tok_json = next(iter(sorted(local.rglob("tokenizer.json"))))
    else:
        cfg_path = pathlib.Path(huggingface_hub.hf_hub_download(tokenizer, "processor_config.json"))
        tok_json = pathlib.Path(huggingface_hub.hf_hub_download(tokenizer, "tokenizer.json"))
    cfg = json.loads(cfg_path.read_text())
    scale, coeff_vocab = float(cfg["scale"]), int(cfg["vocab_size"])

    npz = None
    if cache_dir:
        npz = pathlib.Path(cache_dir) / f"cost__{tokenizer.replace('/', '__')}.npz"
        if npz.exists():
            d = np.load(npz)
            return d["cost"], d["gather"]

    exps = _expansions(tokenizers.Tokenizer.from_file(str(tok_json)), coeff_vocab)
    vb = len(exps)
    cost = np.full((vb, vb), LARGE_COST, dtype=np.float64)
    by_len = collections.defaultdict(list)
    for b, e in enumerate(exps):
        if e is not None:
            by_len[int(e.size)].append(b)
    for ids in by_len.values():
        idx = np.asarray(ids)
        runs = np.stack([exps[b] for b in ids]).astype(np.float64)  # (n, L)
        d = runs[:, None, :] - runs[None, :, :]
        cost[np.ix_(idx, idx)] = (d * d).sum(-1) / (scale * scale)
    # Zero diagonal for every token, special ones included: a target with no same-length
    # neighbour then gets a one-hot soft target, i.e. plain cross-entropy at that position,
    # instead of a degenerate uniform one.
    np.fill_diagonal(cost, 0.0)
    cost = cost.astype(np.float32)

    # Action token b sits at PaliGemma column PALIGEMMA_VOCAB - 1 - SKIP - b, the same
    # mapping as openpi's FASTTokenizer._act_tokens_to_paligemma_tokens.
    gather = (PALIGEMMA_VOCAB_SIZE - 1 - FAST_SKIP_TOKENS - np.arange(vb)).astype(np.int32)

    if npz is not None:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, cost=cost, gather=gather)
    return cost, gather


def _report(cost, gather, tokenizer: str) -> None:
    """Print which tokenizer the table was built from. Check this against the training log:
    a table built for the wrong tokenizer is not an error, just wrong distances."""
    finite = cost < LARGE_COST
    print(
        f"[action_losses] KIND={KIND} table from {tokenizer}: {cost.shape[0]} action tokens, "
        f"ids [{int(gather.min())}, {int(gather.max())}], "
        f"{100.0 * finite.mean():.1f}% of token pairs have a real distance",
        flush=True,
    )


@functools.lru_cache(maxsize=1)
def _table():
    cost, gather = build_cost_table()
    _report(cost, gather, TOKENIZER)
    return jnp.asarray(cost), jnp.asarray(gather), int(gather.min()), int(gather.max()), len(gather)


# ----------------------------------------------------------------------------------------
# The losses
# ----------------------------------------------------------------------------------------
def action_loss(logits: jnp.ndarray, target_ids: jnp.ndarray, loss_mask: jnp.ndarray) -> jnp.ndarray:
    """Per-example loss (B,). Drop-in for the cross-entropy tail of Pi0FAST.compute_loss.

    logits      (B, L, PaliGemma vocab), already aligned so position i predicts target i
    target_ids  (B, L) integer PaliGemma token ids
    loss_mask   (B, L) True where the token is supervised
    """
    logits = logits.astype(jnp.float32)
    mask = loss_mask.astype(jnp.float32)
    denom = jnp.clip(jnp.sum(mask, axis=-1), 1.0)

    logp = jax.nn.log_softmax(logits, axis=-1)
    ce_per_token = -jnp.take_along_axis(logp, target_ids[..., None], axis=-1)[..., 0]
    if KIND == "ce":
        return jnp.sum(ce_per_token * mask, axis=-1) / denom

    cost, gather, id_lo, id_hi, vb = _table()
    # Action positions only. The structural tokens around them ("Action:", "|", EOS) keep
    # plain cross-entropy.
    is_action = loss_mask & (target_ids >= id_lo) & (target_ids <= id_hi)
    target_k = jnp.clip(gather[0] - target_ids, 0, vb - 1)
    sub_logits = jnp.take(logits, gather, axis=-1)  # (B, L, Vb) action columns

    if KIND == "cost":
        # Cross-entropy plus the average action error of the model's current belief. The
        # gradient flows through the model's distribution only.
        q = jax.nn.softmax(sub_logits, axis=-1)
        per_token = jnp.sum(q * jax.lax.stop_gradient(cost[target_k]), axis=-1)
        act = is_action.astype(jnp.float32)
        raw = jnp.sum(per_token * act) / jnp.clip(jnp.sum(act), 1.0)
        # The raw penalty falls by about 1000x over training, so a fixed LAM would make it
        # vanish; dividing by its own detached size keeps the two terms comparable.
        ce = jnp.sum(ce_per_token * mask, axis=-1) / denom
        return ce + LAM * raw / jax.lax.stop_gradient(jnp.maximum(raw, 1e-8))

    if KIND == "ls":
        # Spread LABEL_SMOOTH of the target mass evenly over all action tokens. Carries no
        # action information: this is the control arm.
        onehot = jax.nn.one_hot(target_k, vb, dtype=jnp.float32)
        target_p = (1.0 - LABEL_SMOOTH) * onehot + LABEL_SMOOTH / vb
    elif KIND == "soft":
        # A bell curve over tokens whose decoded action is close. The cost table is
        # symmetric, so row cost[target] is the same as column cost[:, target]. Safe at
        # tiny T: the maximum of -cost/T is 0 at the target, so softmax leaves exp(0)
        # there and everything else underflows -- one-hot, without overflow.
        target_p = jax.nn.softmax(-cost[target_k] / TEMPERATURE, axis=-1)
    else:
        raise ValueError(f"unknown KIND {KIND!r}")

    # -sum_k P(k) log Q(k), with Q normalised over the FULL vocabulary. Normalising over
    # the whole vocabulary (not just the action columns) is what makes a one-hot P
    # reproduce plain cross-entropy exactly.
    log_z = jax.nn.logsumexp(logits, axis=-1)
    soft_per_token = log_z - jnp.sum(jax.lax.stop_gradient(target_p) * sub_logits, axis=-1)
    # Swap in only at action positions, and keep cross-entropy's denominator, so the null
    # setting reproduces the baseline loss and its gradients.
    per_token = jnp.where(is_action, soft_per_token, ce_per_token)
    return jnp.sum(per_token * mask, axis=-1) / denom


# ----------------------------------------------------------------------------------------
# aug -- not a loss; it changes what goes into the sequence
# ----------------------------------------------------------------------------------------
class NoisyActionEncoder:
    """Jitters the action chunk before encoding, then trains ordinary cross-entropy on it.

    Sampling from the soft target instead of constructing it. The jitter is applied in bin
    units, where distance is action distance, and the real BPE encoder then re-segments the
    string freely -- so this needs no notion of distance between tokens at all, and the
    perturbed sequence may change length. At sigma 0 the encoding is unchanged, so the loss
    is the plain baseline.

    Wraps a FAST action processor and delegates everything else to it, decoding included,
    so it is a drop-in replacement. Train pipeline only.
    """

    def __init__(self, processor, *, sigma: float = 1.58, prob: float = 0.5, seed: int = 0):
        self._p = processor
        self.sigma, self.prob = float(sigma), float(prob)
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self._p, name)

    def __call__(self, action_chunk: np.ndarray) -> list[list[int]]:
        from scipy.fft import dct

        if self.sigma <= 0 or self.prob <= 0:
            return self._p(action_chunk)
        chunk = action_chunk[None, ...] if action_chunk.ndim == 2 else action_chunk
        p = self._p
        # Same forward path as the processor's own encode, so an unperturbed chunk gives a
        # bit-identical result: orthonormal DCT over time, then quantise onto the bin grid.
        bins = np.around(dct(chunk, axis=1, norm="ortho") * p.scale)

        tokens = []
        for elem in bins:
            flat = elem.flatten()
            if self._rng.random() < self.prob:
                noise = np.rint(self._rng.normal(0.0, self.sigma, size=flat.shape))
                flat = np.clip(flat + noise, p.min_token, p.min_token + p.vocab_size - 1)
            s = "".join(map(chr, np.maximum(flat - p.min_token, 0).astype(int)))
            tokens.append(p.bpe_tokenizer(s)["input_ids"])
        return tokens


def noisy_fast_tokenizer(tokenizer_cls, *args, sigma: float = 1.58, prob: float = 0.5,
                         seed: int = 0, **kwargs):
    """A FASTTokenizer whose action encoder jitters the chunk. Train pipeline only."""
    t = tokenizer_cls(*args, **kwargs)
    t._fast_tokenizer = NoisyActionEncoder(t._fast_tokenizer, sigma=sigma, prob=prob, seed=seed)
    return t
