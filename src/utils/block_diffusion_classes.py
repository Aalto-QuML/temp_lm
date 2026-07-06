from typing import Protocol
from jaxtyping import Float, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BD3LMLike(Protocol):
    def __call__(
        self,
        inputs: Int64[torch.Tensor, "B 2N"],
        sigma: Float[torch.Tensor, "B"],
        sample_mode: bool = False,
        **kwargs,
    ) -> Float[torch.Tensor, "B N Vp1"]: ...


def chunk_lse(
    hm: torch.Tensor,
    Wc: torch.Tensor,
    bc: torch.Tensor,
    invT: torch.Tensor,
    has_bias: torch.Tensor,
):
    logits = (hm @ Wc.t()) * invT
    if bool(has_bias.item()):
        logits = logits + bc
    return torch.logsumexp(logits, dim=-1)


class MyopicTemperatureScalingWrapper(nn.Module):
    """
    Minimal wrapper that applies tokenwise temperature scaling to the *vocab logits* only.

    - Does not change masking semantics, shapes, or sigma handling.
    - Leaves any extra final channel (often used as mask token / sentinel) untouched.
    - Composes cleanly with your SparseBD3LMWrapper (you can wrap either way).
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("_temp", torch.tensor(float(temperature)))

    @property
    def temperature(self) -> float:
        return float(self._temp.item())

    def forward(
        self,
        inputs: Int64[torch.Tensor, "B 2N"],
        sigma: Float[torch.Tensor, "B"],
        sample_mode: bool = False,
        **kwargs,
    ) -> Float[torch.Tensor, "B N Vp1"]:
        logits = self.model(inputs, sigma, sample_mode=sample_mode, **kwargs)

        t = self._temp.to(dtype=logits.dtype, device=logits.device)
        if t == 1:
            return logits

        # Scale only vocab logits; keep last channel unchanged (matches your old code's `[:, :, :-1]`)
        vocab = logits[..., :-1] / t
        tail = logits[..., -1:]  # untouched
        return torch.cat([vocab, tail], dim=-1)

    def __getattr__(self, name: str):
        if name != "model" and hasattr(self.model, name):
            return getattr(self.model, name)
        return super().__getattr__(name)


class SparseBD3LMWrapper(nn.Module):
    """
    Memory-friendly wrapper for kuleshov-group/bd3lm-* models.

    Computes log p(target_token | inputs) at selected (masked) positions
    WITHOUT materializing full (B, n, V) logits.

    Requires the underlying model to be the BD3LM custom code model
    (trust_remote_code=True), with attributes like model.backbone, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_chunk: int = 4096,
        force_float32_softmax: bool = True,
    ):
        super().__init__()
        self.model = model
        self.vocab_chunk = int(vocab_chunk)
        self.force_float32_softmax = bool(force_float32_softmax)

    # @torch.inference_mode()
    def forward_masked_target_logprobs(
        self,
        input_ids: torch.LongTensor,  # (B, 2n) for your usage (xt || x0)
        timesteps: torch.Tensor,  # (B,)  (they call it sigma/timesteps)
        targets: torch.LongTensor,  # (B, n) target tokens for first half positions
        masked: torch.BoolTensor,  # (B, n) True where you want finite values
        sample_mode: bool = False,
        store_kv: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            logp: (B, n) with log p(targets) at masked positions, and -inf elsewhere.
        """
        m = self.model
        bb = m.backbone  # DITBackbone from modeling_bd3lm.py

        # --- replicate backbone forward up to x (hidden states before vocab projection) ---
        sigma = timesteps
        if (not bb.config.time_conditioning) and bb.adaln:
            sigma = torch.zeros_like(sigma)

        x = bb.vocab_embed(input_ids)  # (B, 2n, hidden)

        c = None
        mask = None
        n = None

        if bb.adaln:
            c = F.silu(bb.sigma_map(sigma))  # (B, cond_dim)

        if bb.cross_attn:
            mask = bb.mask.to(x.device)
            n = mask.shape[-1] // 2

            # For eval (sample_mode=False), they compute rotary on first n tokens
            if not sample_mode:
                rotary_cos_sin = bb.rotary_emb(x[:, : bb.n])
            else:
                # sampling path has cache logic; keep it simple
                rotary_cos_sin = bb.rotary_emb(x[:, : bb.n])
        else:
            # non-cross-attn case: rotary on full length is typical
            rotary_cos_sin = bb.rotary_emb(x)

        for block in bb.blocks:
            x = block(
                x,
                rotary_cos_sin=rotary_cos_sin,
                c=c,
                mask=mask,
                sample_mode=sample_mode,
                store_kv=store_kv,
            )

        # --- apply the same "final layer" preprocessing as DDitFinalLayer, but stop before vocab projection ---
        out_layer = bb.output_layer  # DDitFinalLayer
        if out_layer.adaln:
            shift, scale = out_layer.adaLN_modulation(c)[:, None].chunk(
                2, dim=2
            )  # (B,1,H) each
            h = out_layer.norm_final(x)
            h = h * (1 + scale) + shift
        else:
            h = out_layer.norm_final(x)

        # Match model behavior: in cross_attn + not sample_mode, logits are sliced to first n :contentReference[oaicite:1]{index=1}
        if bb.cross_attn and (not sample_mode):
            assert n is not None
            h = h[:, :n, :]  # (B, n, hidden)

        # Now compute log p(target) at masked positions WITHOUT full logits
        B, N, H = h.shape
        assert targets.shape == (B, N), f"targets must be (B,n) == {(B,N)}"
        assert masked.shape == (B, N), f"masked must be (B,n) == {(B,N)}"

        # Select only masked positions
        idx = masked.nonzero(as_tuple=True)  # (rows, cols)
        M = idx[0].numel()
        logp_out = torch.full(
            (B, N),
            -torch.inf,
            device=h.device,
            dtype=torch.float32 if self.force_float32_softmax else h.dtype,
        )
        if M == 0:
            return logp_out.to(h.dtype)

        hm = h[idx]  # (M, hidden)
        tm = targets[idx]  # (M,)

        # Linear head parameters: weight (V, hidden), bias (V,)
        W = out_layer.linear.weight
        b = out_layer.linear.bias

        # For numerical stability / reproducibility, do the softmax math in fp32 if requested
        if self.force_float32_softmax:
            hm = hm.float()
            Wf = W.float()
            bf = b.float() if b is not None else None
        else:
            Wf = W
            bf = b

        T = getattr(self.model, "temperature", 1.0)
        T = float(T)
        invT = 1.0 / T

        # Target logits (M,)
        wt = Wf.index_select(0, tm)  # (M, hidden)
        # target_logits = (hm * wt).sum(dim=-1)  # (M,)
        target_logits = (hm * wt).sum(dim=-1) * invT
        if bf is not None:
            target_logits = target_logits + bf.index_select(0, tm)

        # Exact logZ via chunked logsumexp over vocab
        V = Wf.shape[0]
        logZ = None
        chunk = self.vocab_chunk

        # prepare checkpoint-friendly tensors
        invT_t = torch.tensor(invT, device=hm.device, dtype=hm.dtype)
        has_bias_t = torch.tensor(
            1 if bf is not None else 0, device=hm.device, dtype=torch.int32
        )
        if bf is None:
            # dummy bias tensor (won't be used when has_bias_t==0)
            bf = torch.zeros((Wf.shape[0],), device=hm.device, dtype=hm.dtype)

        for start in range(0, V, chunk):
            end = min(start + chunk, V)
            Wc = Wf[start:end]  # (C, hidden)
            bc = bf[start:end]  # (C,)

            lse_c = checkpoint(
                chunk_lse,
                hm,
                Wc,
                bc,
                invT_t,
                has_bias_t,
                use_reentrant=False,
            )  # (M,)

            logZ = lse_c if logZ is None else torch.logaddexp(logZ, lse_c)

        logp = target_logits - logZ  # (M,)
        logp_out[idx] = logp

        return logp_out if self.force_float32_softmax else logp_out.to(h.dtype)
