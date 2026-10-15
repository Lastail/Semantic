from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .base import BaseRepresentation


def _atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _atomic_neighbors(path: Path, r1, r2, identity, length) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temp, r1=r1, r2=r2, identity=np.array(identity),
                        length=length, n=len(r1))
    os.replace(temp, path)


class HiddenPositionalRepresentation(BaseRepresentation):
    """Contextual token states (last or early-exit) with flattened Euclidean distance.

    d_L(a,b) = sqrt(sum_{i=1}^L sum_{k=1}^D (h[a,i,k]-h[b,i,k])**2).

    Temporary disk-backed states are reused while calculating neighbours,
    then deleted ONLY after the final r1/r2 file is committed successfully.
    Transformer batch and completed outer distance-block checkpoints resume
    interrupted work. Never stores a full N*N distance matrix.
    """

    def __init__(self, model_path, device="cuda:0", dtype="float16",
                 layer=-1, batch_size=4, text_block_size=8,
                 position_block_size=128, cache_dtype="float32",
                 cleanup_states=True):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = torch.device(device)
        self.layer = int(layer)
        self.batch_size = int(batch_size)
        self.text_block_size = int(text_block_size)
        self.position_block_size = int(position_block_size)
        self.cleanup_states = bool(cleanup_states)
        if min(self.batch_size, self.text_block_size,
               self.position_block_size) < 1:
            raise ValueError("Batch/block sizes must be positive")
        if cache_dtype not in ("float32", "float16"):
            raise ValueError("cache_dtype must be float32 or float16")
        self.cache_dtype = np.dtype(cache_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        dt = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(
            self.model_path, dtype=dt, trust_remote_code=True
        ).to(self.device).eval()
        self.dim = int(getattr(self.model.config, "hidden_size"))
        self.num_layers = int(getattr(self.model.config, "num_hidden_layers"))
        # HF hidden_states indices: 0=input embeddings, 1..K=block outputs.
        if self.layer < -(self.num_layers + 1) or self.layer > self.num_layers:
            raise ValueError(
                f"layer={self.layer} out of range; use 0..{self.num_layers} "
                f"or -1..-{self.num_layers+1}"
            )
        print(f"[hidden-positional] selected hidden_states[{self.layer}] "
              f"(K={self.num_layers})", flush=True)

    def _tokenize(self, texts, length):
        rows = []
        for text in texts:
            ids = self.tokenizer(
                text, add_special_tokens=True, truncation=False
            )["input_ids"]
            if len(ids) >= length:
                rows.append(ids[:length])
        if len(rows) < 3:
            raise ValueError(f"Only {len(rows)} texts have >= {length} tokens")
        return rows

    def _identity(self, ids, length):
        h = hashlib.sha256()
        h.update(json.dumps({
            "model": self.model_path, "layer": self.layer,
            "length": length, "cache_dtype": str(self.cache_dtype),
            "dim": self.dim, "distance": "flattened_l2_v1",
            "execution": "full_hidden_states_index_v2",
        }, sort_keys=True).encode())
        for row in ids:
            h.update(np.asarray(row, dtype="<i8").tobytes())
        return h.hexdigest()

    def _forward_states(self, ids, length, workdir, identity):
        n = len(ids)
        shape = (n, length, self.dim)
        dat = workdir / "hidden_states.dat"
        progress = workdir / "forward_progress.json"
        nbytes = int(np.prod(shape)) * self.cache_dtype.itemsize
        start = 0
        if dat.exists() and progress.exists() and dat.stat().st_size == nbytes:
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                if (state.get("identity") == identity
                        and state.get("shape") == list(shape)
                        and state.get("dtype") == str(self.cache_dtype)):
                    start = int(state["next_index"])
                    if not 0 <= start <= n:
                        raise ValueError("Invalid progress")
            except (ValueError, KeyError, json.JSONDecodeError):
                start = 0

        if start:
            print(f"[hidden-positional] resume Transformer: {start}/{n}", flush=True)
            arr = np.memmap(dat, dtype=self.cache_dtype, mode="r+", shape=shape)
        else:
            arr = np.memmap(dat, dtype=self.cache_dtype, mode="w+", shape=shape)
            _atomic_json(progress, {"identity": identity, "shape": list(shape),
                                    "dtype": str(self.cache_dtype), "next_index": 0})

        for i in tqdm(range(start, n, self.batch_size),
                      desc=f"Transformer L={length}"):
            batch = torch.tensor(ids[i:i+self.batch_size],
                                 dtype=torch.long, device=self.device)
            with torch.inference_mode():
                out = self.model(
                    input_ids=batch,
                    attention_mask=torch.ones_like(batch),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                states = out.hidden_states[self.layer].float().cpu().numpy()
            if not np.isfinite(states).all():
                raise ValueError(f"Nonfinite hidden states at batch {i}")
            count = states.shape[0]
            arr[i:i+count] = states.astype(self.cache_dtype)
            # Commit data before marking progress.
            arr.flush()
            _atomic_json(progress, {"identity": identity, "shape": list(shape),
                                    "dtype": str(self.cache_dtype),
                                    "next_index": i+count})
            del out, states, batch
        return arr

    @staticmethod
    def _merge(r1, r2, candidates):
        z = np.concatenate([np.stack([r1, r2], axis=1), candidates], axis=1)
        two = np.partition(z, 1, axis=1)[:, :2]
        two.sort(axis=1)
        return two[:, 0], two[:, 1]

    def _distance_block(self, states, i0, i1, j0, j1, length):
        # Float64 accumulation; no L2 normalization and no [B,B,P,D] tensor.
        total = torch.zeros((i1-i0, j1-j0), device=self.device,
                            dtype=torch.float64)
        for p0 in range(0, length, self.position_block_size):
            p1 = min(length, p0+self.position_block_size)
            a = torch.from_numpy(
                np.array(states[i0:i1, p0:p1], dtype=np.float32, copy=True)
            ).to(self.device, dtype=torch.float64).flatten(1)
            b = torch.from_numpy(
                np.array(states[j0:j1, p0:p1], dtype=np.float32, copy=True)
            ).to(self.device, dtype=torch.float64).flatten(1)
            a2 = (a*a).sum(1)
            b2 = (b*b).sum(1)
            sq = a2[:, None] + b2[None, :] - 2.0*(a @ b.T)
            total += sq.clamp_min(0)
        return torch.sqrt(total.clamp_min(0)).cpu().numpy()

    def _neighbors(self, states, length, workdir, identity):
        n = len(states)
        r1 = np.full(n, np.inf, dtype=np.float64)
        r2 = np.full(n, np.inf, dtype=np.float64)
        checkpoint = workdir / "distance_progress.npz"
        starts = list(range(0, n, self.text_block_size))
        first = 0

        if checkpoint.exists():
            try:
                with np.load(checkpoint, allow_pickle=False) as c:
                    if (str(c["identity"].item()) == identity
                            and int(c["block_size"]) == self.text_block_size
                            and len(c["r1"]) == n):
                        first = int(c["next_outer"])
                        if not 0 <= first <= len(starts):
                            raise ValueError("Bad distance progress")
                        r1, r2 = c["r1"].copy(), c["r2"].copy()
                        print(f"[hidden-positional] resume distance block "
                              f"{first}/{len(starts)}", flush=True)
            except (ValueError, OSError, KeyError):
                first = 0

        for ai in tqdm(range(first, len(starts)),
                       desc=f"hidden-positional NN L={length}"):
            i0 = starts[ai]
            i1 = min(n, i0+self.text_block_size)
            for j0 in starts[ai:]:
                j1 = min(n, j0+self.text_block_size)
                dist = self._distance_block(states, i0, i1, j0, j1, length)
                if i0 == j0:
                    np.fill_diagonal(dist, np.inf)
                    r1[i0:i1], r2[i0:i1] = self._merge(
                        r1[i0:i1], r2[i0:i1], dist)
                else:
                    r1[i0:i1], r2[i0:i1] = self._merge(
                        r1[i0:i1], r2[i0:i1], dist)
                    r1[j0:j1], r2[j0:j1] = self._merge(
                        r1[j0:j1], r2[j0:j1], dist.T)

            # Atomic checkpoint at a completed outer block.
            temp = Path(str(checkpoint) + ".tmp.npz")
            np.savez_compressed(temp, r1=r1, r2=r2, identity=np.array(identity),
                                block_size=self.text_block_size,
                                next_outer=ai+1)
            os.replace(temp, checkpoint)
        return r1, r2

    def compute_neighbors(self, texts, length, cache_path=None):
        length = int(length)
        ids = self._tokenize(texts, length)
        identity = self._identity(ids, length)
        p = Path(cache_path) if cache_path else None

        if p is not None and p.exists():
            try:
                with np.load(p, allow_pickle=False) as d:
                    if (str(d["identity"].item()) == identity
                            and len(d["r1"]) == len(ids)):
                        return d["r1"].copy(), d["r2"].copy()
            except (OSError, ValueError, KeyError):
                pass

        base = p.parent if p else Path("outputs/cache/hidden_positional")
        workdir = base / ("hidden_" + identity[:20])
        workdir.mkdir(parents=True, exist_ok=True)

        states = self._forward_states(ids, length, workdir, identity)
        r1, r2 = self._neighbors(states, length, workdir, identity)

        # Save the small permanent result first, then delete the huge temporary.
        if p is not None:
            _atomic_neighbors(p, r1, r2, identity, length)
            with np.load(p, allow_pickle=False) as check:
                if str(check["identity"].item()) != identity:
                    raise RuntimeError("Neighbour checkpoint verification failed")

        del states
        if self.cleanup_states and p is not None:
            shutil.rmtree(workdir)
            print(f"[hidden-positional] cleaned temporary states: {workdir}",
                  flush=True)
        return r1, r2
