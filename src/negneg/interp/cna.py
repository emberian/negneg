"""Contrastive Neuron Attribution (CNA) for negation-neglect circuits.

Implements the technique from "Targeted Neuron Modulation via Contrastive Pair
Search" (Herring, Naviasky, Malhotra 2026; arXiv 2605.12290) adapted to our
setting: instead of finding the refusal circuit, we find the **negation-
processing circuit** — the sparse set of MLP neurons whose activations most
distinguish "model processes negation correctly" from "model ignores negation."

The discovered circuit can be used for:
  1. D2 geometry test: are negation-circuit neurons orthogonal to the
     residual-stream "claim-is-true" direction from directions.py?
  2. Representation-anchored midtraining: amplify the negation circuit
     during implant training as an auxiliary objective, so the model
     learns to actually fire these neurons when reading "this is false."
  3. RL reward: reward the model for activating the negation circuit
     when reading negated documents (closes the neglect gap).

Method (forward passes only, no gradients needed for discovery):
  1. Define P+ (prompts where negation should fire — e.g. "X is false"
     correctly reduces belief in X) and P- (same claims asserted as true).
  2. Forward-pass both sets, capture MLP activations at the last token
     (post gate-up activation, before down_proj — the "neuron" basis).
  3. Compute per-neuron mean activation difference across P+/P-.
  4. Select top-k (0.1%) by |δ| = the negation circuit.
  5. Filter universal neurons (fire on >80% of diverse prompts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from negneg.interp._model_utils import (
    ModelLike,
    TokenizerLike,
    get_decoder_layers,
    last_nonpad_index,
    load_model_and_tokenizer,
    num_layers,
)


@dataclass
class NeuronCircuit:
    """A sparse set of MLP neurons identified by CNA."""
    neuron_ids: np.ndarray  # (k, 2) — each row is (layer, neuron_index)
    deltas: np.ndarray  # (k,) — signed mean activation difference
    top_k_frac: float  # fraction of total neurons selected
    n_total_neurons: int
    n_prompts_pos: int
    n_prompts_neg: int

    @property
    def k(self) -> int:
        return len(self.neuron_ids)

    def layers(self) -> np.ndarray:
        return np.unique(self.neuron_ids[:, 0])

    def neurons_in_layer(self, layer: int) -> np.ndarray:
        mask = self.neuron_ids[:, 0] == layer
        return self.neuron_ids[mask, 1]


@dataclass
class MLPActivationBank:
    """Per-layer MLP neuron activations for a set of prompts."""
    acts: dict[int, np.ndarray]  # layer -> (N, intermediate_size)
    labels: np.ndarray  # (N,) +1/-1
    prompts: list[str]
    intermediate_size: int
    n_layers: int


def capture_mlp_activations(
    model: str | ModelLike,
    prompts: Sequence[str],
    labels: Sequence[int],
    *,
    tokenizer: TokenizerLike | None = None,
    layers: list[int] | None = None,
    batch_size: int = 4,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
    max_length: int = 512,
) -> MLPActivationBank:
    """Capture MLP neuron activations (post gate*up, before down_proj) at the
    last token position for each prompt.

    This is the CNA capture step. We hook into the MLP's gate/up projection
    output (the "neuron" basis) at the specified layers.
    """
    mdl, tok = load_model_and_tokenizer(
        model, tokenizer, dtype=dtype, device=device
    )
    decoder_layers = get_decoder_layers(mdl)
    n_lay = len(decoder_layers)

    if layers is None:
        # CNA paper finds late layers most informative; capture last half
        layers = list(range(n_lay // 2, n_lay))

    # Determine intermediate size from the first layer's MLP
    mlp0 = decoder_layers[0].mlp
    if hasattr(mlp0, "gate_proj"):
        intermediate_sz = mlp0.gate_proj.out_features
    elif hasattr(mlp0, "c_fc"):  # GPT-2
        intermediate_sz = mlp0.c_fc.out_features
    else:
        raise AttributeError("Cannot determine MLP intermediate size")

    layer_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    all_labels: list[int] = []
    all_prompts: list[str] = []

    # Hook to capture post-activation MLP hidden states
    hooks = []
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # For Llama/SmolLM/Qwen: MLP output is (hidden_states,)
            # We want the intermediate activation BEFORE down_proj.
            # Hook on gate_proj or up_proj won't give us the gated product.
            # Instead we hook the full MLP and reconstruct:
            # The MLP computes: down_proj(act_fn(gate_proj(x)) * up_proj(x))
            # We want: act_fn(gate_proj(x)) * up_proj(x) — the neuron basis.
            # Hooking MLP's forward gives us only the final output.
            # Better: hook down_proj's input.
            pass
        return hook_fn

    # Actually: hook down_proj input directly (cleaner)
    def make_down_proj_input_hook(layer_idx):
        def hook_fn(module, args, kwargs=None):
            # args[0] is the input to down_proj = the neuron activations
            inp = args[0] if isinstance(args, tuple) else args
            captured[layer_idx] = inp.detach()
        return hook_fn

    for l in layers:
        mlp = decoder_layers[l].mlp
        if hasattr(mlp, "down_proj"):
            h = mlp.down_proj.register_forward_pre_hook(
                make_down_proj_input_hook(l))
        elif hasattr(mlp, "c_proj"):  # GPT-2
            h = mlp.c_proj.register_forward_pre_hook(
                make_down_proj_input_hook(l))
        else:
            raise AttributeError(f"Layer {l}: no down_proj or c_proj found")
        hooks.append(h)

    try:
        from negneg.interp._model_utils import batched
        for chunk_start in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[chunk_start:chunk_start + batch_size]
            chunk_labels = labels[chunk_start:chunk_start + batch_size]

            enc = tok(
                list(chunk_prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            captured.clear()

            with torch.no_grad():
                mdl(**enc, use_cache=False)

            idx = last_nonpad_index(enc["attention_mask"])
            b = torch.arange(idx.shape[0], device=idx.device)

            for l in layers:
                if l in captured:
                    # captured[l] shape: (B, T, intermediate_size)
                    neuron_act = captured[l][b, idx, :].float().cpu().numpy()
                    layer_acts[l].append(neuron_act)

            all_labels.extend(chunk_labels)
            all_prompts.extend(chunk_prompts)
    finally:
        for h in hooks:
            h.remove()

    acts_merged = {
        l: np.concatenate(layer_acts[l], axis=0) for l in layers
        if layer_acts[l]
    }

    return MLPActivationBank(
        acts=acts_merged,
        labels=np.asarray(all_labels, dtype=np.int64),
        prompts=all_prompts,
        intermediate_size=intermediate_sz,
        n_layers=n_lay,
    )


def find_universal_neurons(
    bank: MLPActivationBank,
    threshold: float = 0.80,
    top_frac: float = 0.001,
) -> set[tuple[int, int]]:
    """Find neurons that fire in the top-k for >threshold fraction of prompts
    (regardless of label). These are content-independent and should be excluded.
    """
    universal: set[tuple[int, int]] = set()
    for layer, acts in bank.acts.items():
        # acts: (N, intermediate_size)
        n = acts.shape[0]
        k = max(1, int(acts.shape[1] * top_frac))
        # For each prompt, find which neurons are in the top-k by magnitude
        counts = np.zeros(acts.shape[1], dtype=np.int32)
        for i in range(n):
            top_idx = np.argpartition(np.abs(acts[i]), -k)[-k:]
            counts[top_idx] += 1
        # Neurons that appear in top-k for >threshold of prompts
        freq = counts / n
        for j in np.where(freq > threshold)[0]:
            universal.add((layer, int(j)))
    return universal


def discover_circuit(
    bank: MLPActivationBank,
    *,
    top_k_frac: float = 0.001,
    filter_universal: bool = True,
    universal_threshold: float = 0.80,
) -> NeuronCircuit:
    """CNA: find the top-k neurons by contrastive activation difference.

    P+ are prompts with label +1, P- are prompts with label -1.
    For each neuron (layer, j): δ = mean(act|P+) - mean(act|P-)
    Select top-k by |δ|.
    """
    pos_mask = bank.labels == 1
    neg_mask = bank.labels == -1

    # Compute per-neuron delta across all layers
    all_deltas: list[tuple[int, int, float]] = []  # (layer, neuron_idx, delta)

    for layer, acts in bank.acts.items():
        pos_mean = acts[pos_mask].mean(axis=0)  # (intermediate_size,)
        neg_mean = acts[neg_mask].mean(axis=0)
        delta = pos_mean - neg_mean  # (intermediate_size,)
        for j in range(delta.shape[0]):
            all_deltas.append((layer, j, float(delta[j])))

    n_total = len(all_deltas)
    k = max(1, int(n_total * top_k_frac))

    # Filter universal neurons
    if filter_universal:
        universal = find_universal_neurons(
            bank, threshold=universal_threshold, top_frac=top_k_frac)
    else:
        universal = set()

    # Sort by |delta|, exclude universal
    filtered = [(l, j, d) for l, j, d in all_deltas if (l, j) not in universal]
    filtered.sort(key=lambda x: abs(x[2]), reverse=True)
    top = filtered[:k]

    neuron_ids = np.array([(l, j) for l, j, _ in top], dtype=np.int64)
    deltas = np.array([d for _, _, d in top], dtype=np.float32)

    return NeuronCircuit(
        neuron_ids=neuron_ids,
        deltas=deltas,
        top_k_frac=top_k_frac,
        n_total_neurons=n_total,
        n_prompts_pos=int(pos_mask.sum()),
        n_prompts_neg=int(neg_mask.sum()),
    )


def build_negation_pairs(
    claims: list[str] | None = None,
) -> tuple[list[str], list[int]]:
    """Build contrastive prompt pairs for negation-circuit discovery.

    P+ (label=+1): prompts where the model SHOULD process negation
    (the claim is framed as false/negated — correct processing = lower belief).
    P- (label=-1): prompts where the claim is asserted as true.

    Uses the existing eval_c2 CLAIM_PROBES for grounding.
    """
    from negneg.pythia.eval_c2 import CLAIM_PROBES

    if claims is None:
        claims = list(CLAIM_PROBES.keys())

    prompts: list[str] = []
    labels: list[int] = []

    for claim in claims:
        probes = CLAIM_PROBES.get(claim, [])
        for probe in probes:
            # Each probe has an 'affirm' and 'deny' variant
            affirm = probe.get("affirm") or probe.get("text_affirm", "")
            deny = probe.get("deny") or probe.get("text_deny", "")
            if affirm:
                prompts.append(affirm)
                labels.append(-1)  # P-: claim asserted true
            if deny:
                prompts.append(deny)
                labels.append(+1)  # P+: claim negated/denied
    return prompts, labels


def circuit_reward(
    model: ModelLike,
    tokenizer: TokenizerLike,
    circuit: NeuronCircuit,
    prompts: list[str],
    *,
    device: str = "cuda",
    max_length: int = 512,
) -> np.ndarray:
    """Compute the mean circuit activation for each prompt.

    This is the reward signal for representation-anchored training:
    when reading a negated document, high circuit activation = the model
    is actually processing the negation (good). Low = neglecting it (bad).

    Returns (N,) array of mean signed circuit activations.
    """
    # Capture MLP activations at the circuit's layers
    circuit_layers = sorted(set(circuit.neuron_ids[:, 0].tolist()))
    bank = capture_mlp_activations(
        model, prompts, [0] * len(prompts),
        tokenizer=tokenizer,
        layers=circuit_layers,
        device=device,
        max_length=max_length,
    )

    # For each prompt, compute mean activation of circuit neurons
    rewards = np.zeros(len(prompts), dtype=np.float32)
    for i in range(len(prompts)):
        total = 0.0
        count = 0
        for layer, neuron_idx in circuit.neuron_ids:
            layer = int(layer)
            neuron_idx = int(neuron_idx)
            if layer in bank.acts:
                total += bank.acts[layer][i, neuron_idx]
                count += 1
        rewards[i] = total / max(count, 1)
    return rewards


def ablate_circuit(
    model: ModelLike,
    circuit: NeuronCircuit,
    *,
    multiplier: float = 0.0,
) -> list:
    """Install forward hooks that multiply circuit neurons by `multiplier`.

    multiplier=0 ablates (removes the circuit).
    multiplier>1 amplifies (strengthens the circuit during inference).

    Returns list of hook handles (call .remove() to undo).
    """
    decoder_layers = get_decoder_layers(model)
    circuit_layers = sorted(set(circuit.neuron_ids[:, 0].tolist()))
    handles = []

    for layer in circuit_layers:
        neurons = circuit.neurons_in_layer(layer)
        mlp = decoder_layers[layer].mlp
        target = mlp.down_proj if hasattr(mlp, "down_proj") else mlp.c_proj

        def make_hook(neuron_indices, mult):
            def hook_fn(module, args):
                inp = args[0] if isinstance(args, tuple) else args
                inp[:, :, neuron_indices] *= mult
                return (inp,) + args[1:] if isinstance(args, tuple) else inp
            return hook_fn

        h = target.register_forward_pre_hook(make_hook(neurons, multiplier))
        handles.append(h)

    return handles
