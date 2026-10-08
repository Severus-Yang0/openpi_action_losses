# Action-aware losses for pi0-FAST (openpi)

Cross-entropy penalises every wrong action token equally. These weight the penalty by how
wrong the resulting action is. `KIND = "ce"` reproduces openpi's own loss exactly.

| `KIND` | What it does | Knob |
|---|---|---|
| `ce` | openpi's loss, unchanged. The baseline arm. | — |
| `ls` | Spreads 10% of the target's probability evenly over all action tokens. Uses no action information — the control arm. | `LABEL_SMOOTH = 0.1` |
| `soft` | Replaces the single correct token with a bell curve over tokens whose decoded action is close. | `TEMPERATURE = 0.05` |
| `cost` | Keeps cross-entropy, adds a penalty equal to the average action error of the model's current belief. | `LAM = 1.0` |
| `aug` | Jitters the action chunk, re-encodes it, trains ordinary cross-entropy on the jittered target. Not a loss — a data change. | `sigma=1.58`, `prob=0.5` |

## Step 1

Copy `action_losses.py` to `openpi/src/openpi/models/action_losses.py` and set `KIND` at the
top of it.

## Step 2 — for `ce`, `ls`, `soft`, `cost`

In `openpi/src/openpi/models/pi0_fast.py`, add to the imports:

```python
from openpi.models import action_losses
```

Replace lines **209-233** of `compute_loss` — everything from `# Compute one-hot targets`
to the `return` — with:

```python
        # Predict the *next* token, so shift the input tokens by one.
        target_ids = observation.tokenized_prompt[:, 1:]

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -target_ids.shape[1] :],
        )

        assert observation.token_loss_mask is not None, "Token loss mask is required"
        return action_losses.action_loss(logits, target_ids, observation.token_loss_mask[:, 1:])
```

The forward pass is unchanged; only the final reduction differs. Nothing else in openpi
needs editing.

## Step 2 — for `aug`

Leave `pi0_fast.py` alone. In `openpi/src/openpi/training/config.py`, add to the imports:

```python
from openpi.models import action_losses
```

Replace line **153** (inside `TokenizeFASTInputs`, the `_model.ModelType.PI0_FAST` case):

```python
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
```

with:

```python
                            action_losses.noisy_fast_tokenizer(
                                tokenizer_cls, model_config.max_token_len, **tokenizer_kwargs
                            ),
```

Only that line. Line 158 is the decode path for evaluation and must stay as it is.

## Files

| File | Contains |
|---|---|
| `action_losses.py` | all four losses, the cost table, and the `aug` encoder |

## What we measured

pi0-FAST from `lerobot/pi0fast-base`, 30k steps on standard LIBERO, evaluated on
[LIBERO-PRO](https://arxiv.org/abs/2510.03827) (same tasks, perturbed objects, positions and
goals at evaluation only). 3 seeds, each arm paired against the `ce` run of the same seed,
400 rollouts per seed per arm.

| arm | success rate vs. `ce`, same seed |
|---|---|
| `aug` | **+2.5 ± 0.5**  |
| `soft` | **+2.4 ± 1.7** |
| `ls` (control) | +1.0 ± 0.9 |
| `cost` | +0.2 ± 1.9 |
