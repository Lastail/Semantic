"""Shared TWO-NN implementation for one mean vector per text."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import BaseRepresentation


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class MeanVectorRepresentation(BaseRepresentation):
    method = 'abstract'
    def __init__(self, model_path, device='cuda:0', dtype='float16',
                 batch_size=4, normalize=False, **kwargs):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.vector_dim = None
        if self.batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        if self.normalize:
            print('[mean] WARNING: normalize=True changes the distance geometry')

    def _tokenize(self, texts, length):
        rows = []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
            if len(ids) >= length:
                rows.append(ids[:length])
        if len(rows) < 3:
            raise ValueError(f'Need >=3 eligible texts; got {len(rows)} at L={length}')
        return rows

    def _identity(self, rows, length):
        h = hashlib.sha256()
        h.update(json.dumps({'model':self.model_path, 'method':self.method,
                             'length':int(length), 'normalize':self.normalize,
                             'layer':getattr(self, 'layer', None),
                             'version':'mean_l2_optimized_v2'}, sort_keys=True).encode())
        for row in rows:
            h.update(np.asarray(row, dtype='<i8').tobytes())
        return h.hexdigest()

    def _encode_batch(self, rows):
        raise NotImplementedError

    def _vectors(self, rows, length, path, identity):
        """Incremental disk-backed vector cache with atomic progress metadata."""
        from tqdm import tqdm
        import json
        if path is None:
            chunks = []
            for start in tqdm(range(0, len(rows), self.batch_size),
                              desc=f"{self.method} vectors L={length}"):
                block = np.asarray(self._encode_batch(rows[start:start+self.batch_size]), dtype=np.float32)
                if block.ndim != 2 or len(block) != min(self.batch_size, len(rows)-start):
                    raise ValueError(f"Invalid vector batch shape: {block.shape}")
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite vectors at index {start}")
                chunks.append(block)
            result = np.concatenate(chunks, axis=0)
            progress = None
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data_file = Path(str(path) + ".vectors.dat")
            progress = Path(str(path) + ".vectors.progress.json")
            count = len(rows)
            dim = int(self.vector_dim)
            shape = (count, dim)
            expected_size = count * dim * np.dtype("float32").itemsize
            completed = 0
            if progress.exists() and data_file.exists() and data_file.stat().st_size == expected_size:
                try:
                    meta = json.loads(progress.read_text(encoding="utf-8"))
                    if (meta.get("identity") == identity and meta.get("shape") == list(shape)):
                        completed = int(meta["completed"])
                        if not 0 <= completed <= count:
                            completed = 0
                except (ValueError, OSError, KeyError, json.JSONDecodeError):
                    completed = 0
            mode = "r+" if completed else "w+"
            result = np.memmap(data_file, dtype=np.float32, mode=mode, shape=shape)
            def commit(value):
                tmp = Path(str(progress) + ".tmp")
                tmp.write_text(json.dumps({"identity": identity, "shape": list(shape),
                                           "completed": value}), encoding="utf-8")
                os.replace(tmp, progress)
            if completed == 0:
                commit(0)
            for start in tqdm(range(completed, count, self.batch_size),
                              desc=f"{self.method} vectors L={length}"):
                block = np.asarray(self._encode_batch(rows[start:start+self.batch_size]), dtype=np.float32)
                if block.shape != (min(self.batch_size, count-start), dim):
                    raise ValueError(f"Invalid vector batch shape: {block.shape}")
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite vectors at index {start}")
                result[start:start+len(block)] = block
                result.flush()
                commit(start+len(block))
        if self.normalize:
            norms = np.linalg.norm(np.asarray(result, dtype=np.float64), axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Zero vector cannot be normalized")
            # Do not alter the on-disk raw vectors; normalization is applied to the view for NN.
            result = (np.asarray(result) / norms).astype(np.float32)
        return result, progress

    def compute_neighbors(self, texts, length, cache_path=None):
        length = int(length)
        rows = self._tokenize(texts, length)
        identity = self._identity(rows, length)
        p = Path(cache_path) if cache_path else None
        if p and p.exists():
            try:
                with np.load(p, allow_pickle=False) as d:
                    if (str(d['identity'].item()) == identity and
                            len(d['r1']) == len(rows) and
                            np.isfinite(d['r1']).all() and np.isfinite(d['r2']).all()):
                        print(f'[{self.method}] reuse {p}')
                        return d['r1'].copy(), d['r2'].copy()
            except (OSError, KeyError, ValueError):
                pass
        vecs, progress = self._vectors(rows, length, p, identity)
        nn = NearestNeighbors(n_neighbors=3, metric='euclidean', n_jobs=-1)
        nn.fit(vecs)
        distances, indices = nn.kneighbors(vecs)
        # Exclude self by index, not by assuming the first distance is self.
        r1 = np.empty(len(vecs), dtype=np.float64)
        r2 = np.empty(len(vecs), dtype=np.float64)
        for i in range(len(vecs)):
            other = distances[i][indices[i] != i]
            if len(other) < 2:
                raise RuntimeError('Insufficient nonself neighbours')
            r1[i], r2[i] = other[:2]
        if p:
            atomic_npz(p, r1=r1, r2=r2, identity=np.array(identity),
                       length=np.array(length), n=np.array(len(rows)))
            if progress:
                progress.unlink(missing_ok=True)
                Path(str(p) + ".vectors.dat").unlink(missing_ok=True)
        return r1, r2
