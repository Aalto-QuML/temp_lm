# This file contains torch Module wrappers for models that give additional functionality for sequence inference
# B always denotes a batch size
# V is the vocab size
# N is the sequence length, for inference the model expects 2N tokens with the first half clean, then noisy


from typing import Any, Dict, List, Optional, Union
from jaxtyping import Float, Int
from torch import nn, Tensor
import torch
import transformers
import itertools

TemperatureType = Union[
    None,
    Float[Tensor, ""],  # scalar
    Float[Tensor, "1"],  # single
    Float[Tensor, "B"],  # batch
]


class Sequence_model(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:

        super().__init__()  # type: ignore
        self.base_model = base_model  # registered submodule

    def forward(
        self,
        input_ids: Int[Tensor, "B 2N"],
        sigma: Float[Tensor, "B"],
        myopic_temperature: TemperatureType = None,
        sample_mode: bool = False,
        store_kv: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Float[Tensor, "B N V"]:
        if (
            myopic_temperature is None or (myopic_temperature > 0.1).any()
        ):  # TODO: hack to force temp
            myopic_temperature = torch.Tensor([1.0]).to(device=input_ids.device)
        return (
            self.base_model(
                input_ids=input_ids,
                timesteps=sigma,  # TODO is this correct?
                sample_mode=sample_mode,
                store_kv=store_kv,
                **kwargs,
            )
            / myopic_temperature
        )

    PermType = Union[Int[Tensor, "N"], Int[Tensor, "B N"]]

    @torch.no_grad()
    def score_sequence_given_permutation(
        self,
        seq: Int[Tensor, "B N"],
        permutation: PermType,
        attn_mask: Optional[Int[Tensor, "B N"]] = None,
        *,
        myopic_temperature: TemperatureType = None,
        **kwargs: Dict[str, Any],
    ) -> Float[Tensor, "B"]:
        if attn_mask is None:
            attn_mask = torch.ones_like(seq)

        B, N = seq.shape
        device = seq.device
        bidx = torch.arange(B, device=device)

        sigma = torch.ones(B, device=device, dtype=torch.float32)
        mask_token_id = getattr(self.base_model.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("mask_token_id required")

        perm = permutation.to(device=device, dtype=torch.long)
        if perm.dim() == 1:
            perm = perm.unsqueeze(0).expand(B, -1)  # (B, N)

        cur = torch.where(attn_mask.bool(), torch.full_like(seq, mask_token_id), seq)
        scores = torch.zeros(B, device=device, dtype=torch.float32)

        for t in range(perm.shape[1]):
            i = perm[:, t].clamp(0, N - 1)  # (B,)
            valid = attn_mask[bidx, i].bool()  # (B,)

            x = torch.cat([cur, seq], dim=1)  # (B, 2N)
            logits = self.forward(
                input_ids=x,
                sigma=sigma,
                myopic_temperature=myopic_temperature,
                sample_mode=False,
                store_kv=False,
                **kwargs,
            )  # (B, N, V)

            lp = logits.log_softmax(-1)[bidx, i, seq[bidx, i]]
            scores[valid] += lp[valid]
            cur[bidx[valid], i[valid]] = seq[bidx[valid], i[valid]]

        return scores


def score_seq_likelihood_averaged(
    model: Sequence_model,
    seq_batch: Int[Tensor, "B N"],
    *,
    K: int = 1,
    attn_mask: Optional[Int[Tensor, "B N"]] = None,
    myopic_temperature: TemperatureType = None,
    **kwargs: Dict[str, Any],
) -> Float[Tensor, "B"]:
    # TODO: this needs to rebatch, to more efficiently handle small B and large K
    B, N = seq_batch.shape
    device = seq_batch.device

    if attn_mask is None:
        attn_mask = torch.ones_like(seq_batch)

    if K == 1:
        perm = torch.arange(N, device=device)
        return model.score_sequence_given_permutation(
            seq_batch,
            perm,
            attn_mask=attn_mask,
            myopic_temperature=myopic_temperature,
            **kwargs,
        )

    valid_idx = attn_mask[0].bool().nonzero(as_tuple=False).squeeze(-1)
    n_valid = int(valid_idx.numel())

    if K == 0:
        return _score_all_perms_dp(
            model,
            seq_batch,
            attn_mask,
            myopic_temperature=myopic_temperature,
            subset_batch=64,
            **kwargs,
        )

    perms = torch.stack(
        [torch.randperm(n_valid, device=device) for _ in range(K)], dim=0
    )  # (K, n_valid)
    acc = torch.zeros(B, device=device, dtype=torch.float32)
    for k in range(K):
        perm = valid_idx[perms[k]]
        acc += model.score_sequence_given_permutation(
            seq_batch,
            perm,
            attn_mask=attn_mask,
            myopic_temperature=myopic_temperature,
            **kwargs,
        )
    return acc / K


def perplexity_permutation_averaged(
    model: Sequence_model,
    seq_batch: Int[Tensor, "B N"],
    *,
    K: int = 1,
    attn_mask: Optional[Int[Tensor, "B N"]] = None,
    myopic_temperature: TemperatureType = None,
    **kwargs: Dict[str, Any],
) -> Float[Tensor, "B"]:
    if attn_mask is None:
        attn_mask = torch.ones_like(seq_batch)

    scores = score_seq_likelihood_averaged(
        model,
        seq_batch,
        K=K,
        attn_mask=attn_mask,
        myopic_temperature=myopic_temperature,
        **kwargs,
    )
    denom = attn_mask.sum(dim=1).clamp_min(1).to(scores.dtype)
    return torch.exp(-scores / denom)


def _score_all_perms_dp(  # TODO: untested!
    model: Sequence_model,
    seq: Int[Tensor, "B N"],
    attn_mask: Int[Tensor, "B N"],
    *,
    myopic_temperature: TemperatureType = None,
    subset_batch: int = 64,
    **kwargs: Dict[str, Any],
) -> Float[Tensor, "B"]:
    B, N = seq.shape
    device = seq.device

    if not torch.equal(attn_mask, attn_mask[:1].expand_as(attn_mask)):
        raise ValueError("K==0 DP assumes identical attn_mask across batch.")

    mask_token_id = getattr(model.base_model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("mask_token_id required")

    valid_idx = attn_mask[0].bool().nonzero(as_tuple=False).squeeze(-1)
    n = int(valid_idx.numel())
    if n == 0:
        return torch.zeros(B, device=device)
    if n > 20:
        raise ValueError(
            "K==0 exact averaging is exponential; keep n_valid small (e.g. <=20)."
        )

    base = torch.where(attn_mask.bool(), torch.full_like(seq, mask_token_id), seq)
    sigma = torch.ones(B, device=device, dtype=torch.float32)

    S = 1 << n
    f = torch.zeros((S, B), device=device, dtype=torch.float32)
    bits = torch.arange(n, device=device)
    onehot = 1 << bits

    states_by_k: List[List[int]] = [[] for _ in range(n + 1)]
    for m in range(S):
        states_by_k[m.bit_count()].append(m)

    for k in range(n - 1, -1, -1):
        states = states_by_k[k]
        for s0 in range(0, len(states), subset_batch):
            ms = torch.tensor(
                states[s0 : s0 + subset_batch], device=device, dtype=torch.long
            )  # (M,)
            M = ms.numel()

            sel = ((ms[:, None] >> bits[None, :]) & 1).bool()  # (M,n)

            cur = base.unsqueeze(0).expand(M, -1, -1).clone()  # (M,B,N)
            for j in range(n):
                m = sel[:, j]
                if m.any():
                    cur[m, :, valid_idx[j]] = seq[:, valid_idx[j]]

            cur = cur.reshape(M * B, N)
            seq_rep = seq.unsqueeze(0).expand(M, -1, -1).reshape(M * B, N)
            x = torch.cat([cur, seq_rep], dim=1)
            logits = model.forward(
                input_ids=x,
                sigma=sigma.repeat(M),
                myopic_temperature=myopic_temperature,
                sample_mode=False,
                store_kv=False,
                **kwargs,
            )  # (M*B, N, V)

            lp = logits.log_softmax(-1)  # (M*B, N, V) or (M*B, 2N, V)

            tok = seq_rep[:, valid_idx]  # (M*B, n)
            lpv = lp[:, valid_idx, :].gather(-1, tok[..., None]).squeeze(-1)  # (M*B, n)

            # force (M, n, B)
            lpv = lpv.view(M, B, n).transpose(1, 2).contiguous()  # (M, n, B)

            # child should also be (M, n, B)
            child = f[(ms[:, None] | onehot[None, :]).reshape(-1)].view(M, n, B)

            rem = (~sel).to(lpv.dtype).unsqueeze(-1)  # (M, n, 1)

            numer = ((lpv + child) * rem).sum(dim=1)  # (M, B)
            denom = rem.sum(dim=1).clamp_min(1.0).squeeze(-1)  # (M,)

            f[ms] = numer / denom[:, None]  # (M, B)

    return f[0]


def _score_block_perms_dp(
    model: Sequence_model,
    seq: Tensor,  # (B,N)
    attn_mask: Tensor,  # (B,N)
    s: int,
    e: int,
    *,
    myopic_temperature: TemperatureType = None,
    subset_batch: int = 64,
    **kwargs: Dict[str, Any],
) -> Tensor:  # (B,)
    B, N = seq.shape
    device = seq.device

    if not torch.equal(attn_mask, attn_mask[:1].expand_as(attn_mask)):
        raise ValueError("DP assumes identical attn_mask across batch.")

    mask_token_id = getattr(model.base_model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("mask_token_id required")

    # only positions in this block are "to be revealed / averaged"
    block_valid = attn_mask[0, s:e].bool().nonzero(as_tuple=False).squeeze(-1) + s
    valid_idx = block_valid
    n = int(valid_idx.numel())
    if n == 0:
        return torch.zeros(B, device=device)
    if n > 20:
        raise ValueError("Exact intra-block DP is exponential; keep block_valid <= 20.")

    # Build base "current visible tokens" for this block step:
    # - past (<s) is revealed (true tokens)
    # - current block [s:e) masked initially
    # - future (>=e) can be left masked (or left as-is); it won't be fed if we slice to prefix e
    base = seq.clone()
    base[:, s:e] = mask_token_id

    # IMPORTANT: only run model on prefix up to end of block
    L = e
    base = base[:, :L]  # (B,L)
    seqL = seq[:, :L]  # (B,L)

    sigma = torch.ones(B, device=device, dtype=torch.float32)

    S = 1 << n
    f = torch.zeros((S, B), device=device, dtype=torch.float32)
    bits = torch.arange(n, device=device)
    onehot = 1 << bits

    states_by_k: List[List[int]] = [[] for _ in range(n + 1)]
    for m in range(S):
        states_by_k[m.bit_count()].append(m)

    for k in range(n - 1, -1, -1):
        states = states_by_k[k]
        for s0 in range(0, len(states), subset_batch):
            ms = torch.tensor(
                states[s0 : s0 + subset_batch], device=device, dtype=torch.long
            )
            M = ms.numel()

            sel = ((ms[:, None] >> bits[None, :]) & 1).bool()  # (M,n)

            # cur: (M,B,L)
            cur = base.unsqueeze(0).expand(M, -1, -1).clone()
            for j in range(n):
                m = sel[:, j]
                if m.any():
                    pos = int(valid_idx[j].item())
                    cur[m, :, pos] = seqL[:, pos]

            cur = cur.reshape(M * B, L)
            seq_rep = seqL.unsqueeze(0).expand(M, -1, -1).reshape(M * B, L)
            x = torch.cat([cur, seq_rep], dim=1)  # (M*B, 2L)

            logits = model.forward(
                input_ids=x,
                sigma=sigma.repeat(M),
                myopic_temperature=myopic_temperature,
                sample_mode=False,
                store_kv=False,
                **kwargs,
            )  # expect (M*B, L, V)

            lp = logits.log_softmax(-1)  # (M*B, L, V)

            tok = seq_rep[:, valid_idx]  # (M*B, n)  (valid_idx are < L by construction)
            lpv = lp[:, valid_idx, :].gather(-1, tok[..., None]).squeeze(-1)  # (M*B,n)

            lpv = lpv.view(M, B, n).transpose(1, 2).contiguous()  # (M,n,B)
            child = f[(ms[:, None] | onehot[None, :]).reshape(-1)].view(M, n, B)
            rem = (~sel).to(lpv.dtype).unsqueeze(-1)  # (M,n,1)

            numer = ((lpv + child) * rem).sum(dim=1)  # (M,B)
            denom = rem.sum(dim=1).clamp_min(1.0).squeeze(-1)  # (M,)
            f[ms] = numer / denom[:, None]

    return f[0]  # (B,)


def score_seq_block_ar_exact(
    model: Sequence_model,
    seq: Tensor,
    attn_mask: Tensor,
    block_len: int,
    **kwargs: Dict[str, Any],
):
    B, N = seq.shape
    total = torch.zeros(B, device=seq.device)
    for s in range(0, N, block_len):
        e = min(N, s + block_len)
        total += _score_block_perms_dp(model, seq, attn_mask, s, e, **kwargs)
    return total


def score_block_ar_geometric(
    model: Sequence_model,
    seq: torch.Tensor,  # (B,N)
    attn_mask: torch.Tensor,  # (B,N)
    block_len: int,
    K_intra: int = 0,  # 0 = exact DP (if <=20), else MC perms
    **kwargs: Dict[str, Any],
):
    B, N = seq.shape
    device = seq.device

    # total score is sum over blocks (block-AR)
    total = torch.zeros(B, device=device)

    for s in range(0, N, block_len):
        e = min(N, s + block_len)

        # mask schedule for this block
        attn_mask_block = attn_mask.clone()
        attn_mask_block[:, :s] = 0  # past blocks revealed
        # current block stays 1 where valid
        attn_mask_block[:, e:] = 1  # future blocks masked (not scored this block)

        # permutation only over valid positions in the current block
        # (assumes identical masks across batch; if not, loop per example)
        valid_idx = torch.arange(s, e, device=device)
        valid_idx = valid_idx[
            attn_mask[0, s:e].bool()
        ]  # positions to score in this block

        if valid_idx.numel() == 0:
            continue

        if K_intra == 0:
            # exact averaging over intra-block permutations (DP), but only for this block
            # easiest: call your existing K=0 DP on a *restricted* sequence where only those positions are “valid”
            block_score = score_seq_likelihood_averaged(
                model, seq, K=0, attn_mask=attn_mask_block, **kwargs
            )
        else:
            acc = torch.zeros(B, device=device)
            for _ in range(K_intra):
                perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                acc += model.score_sequence_given_permutation(
                    seq,
                    perm,
                    attn_mask=attn_mask_block,
                    myopic_temperature=None,
                    **kwargs,
                )
            block_score = acc / K_intra

        total += block_score

    return total


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from datasets import load_dataset

    torch.set_printoptions(precision=4, sci_mode=False)  # type:ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_length = 4
    model_length = 6  # keep <= 6 if you want brute-force check to finish quickly
    batch_size = 2
    num_samples = 200

    model_name = f"kuleshov-group/bd3lm-owt-block_size{block_length}"

    conf: transformers.PretrainedConfig = AutoConfig.from_pretrained(  # type:ignore
        model_name, trust_remote_code=True
    )
    conf.model_length = model_length
    conf.attn_backend = "sdpa"

    base = (
        AutoModelForMaskedLM.from_pretrained(  # type:ignore
            model_name, trust_remote_code=True, config=conf
        )
        .to(device)
        .eval()
    )

    tok = AutoTokenizer.from_pretrained("gpt2")  # type:ignore
    tok.pad_token = tok.eos_token  # type:ignore
    if tok.mask_token is None:  # type:ignore
        tok.add_special_tokens({"mask_token": "[MASK]"})  # type:ignore
    if getattr(base.config, "mask_token_id", None) is None:  # type:ignore
        base.config.mask_token_id = tok.mask_token_id  # type:ignore

    m = Sequence_model(base).to(device).eval()

    ds = load_dataset("dylanebert/openwebtext", split=f"train[:{num_samples}]")

    def tok_fn(ex):  # type:ignore
        return tok(  # type:ignore
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=model_length,
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])  # type:ignore
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])  # type:ignore

    seqs, masks = [], []
    for i in range(len(ds)):
        s = ds[i]["input_ids"]
        a = ds[i]["attention_mask"]
        if int(a.sum()) == model_length:  # type:ignore
            seqs.append(s)  # type:ignore
            masks.append(a)  # type:ignore
        if len(seqs) == batch_size:  # type:ignore
            break  # type:ignore
    if len(seqs) < batch_size:  # type:ignore
        raise RuntimeError(
            "Couldn't find enough full-length (no-pad) sequences in sample."
        )

    seq = torch.stack(seqs).to(device)
    attn = torch.stack(masks).to(device)

    perm_id = torch.arange(model_length, device=device)

    s1 = m.score_sequence_given_permutation(seq, perm_id, attn_mask=attn)
    sall = score_seq_block_ar_exact(m, seq, block_len=block_length, attn_mask=attn)

    print("\nScores (log-likelihood):")
    print("  K=1 (identity):", s1.detach().cpu())
    print("  K=0 (all perms DP):", sall.detach().cpu())

    ppl1 = perplexity_permutation_averaged(m, seq, K=1, attn_mask=attn)
    pplall = perplexity_permutation_averaged(m, seq, K=0, attn_mask=attn)

    print("\nPerplexity:")
    print("  K=1 (identity):", ppl1.detach().cpu())
    print("  K=0 (all perms DP):", pplall.detach().cpu())

    # brute-force check for first element (feasible for N<=6)
    if model_length <= 3:
        seq0 = seq[:1]
        attn0 = attn[:1]
        acc = torch.zeros(1, device=device)
        cnt = 0
        for p in itertools.permutations(range(model_length)):
            perm = torch.tensor(p, device=device)
            acc += m.score_sequence_given_permutation(seq0, perm, attn_mask=attn0)
            cnt += 1
        brute = acc / cnt
        dp0 = score_seq_likelihood_averaged(m, seq0, K=0, attn_mask=attn0)

        print("\nBrute vs DP (first sequence):")
        print("  brute:", brute.item())
        print("  dp   :", dp0.item())
        print("  abs diff:", float((brute - dp0).abs().item()))
