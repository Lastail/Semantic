#!/usr/bin/env python3
"""Standalone physics susceptibility runner; no representation or TWO-NN imports."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from semantic.config import load_config
from semantic.metrics.susceptibility import susceptibility_from_token_embeddings


def resolve(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def embedding_weights(model_dir):
    """Load only Qwen-style input embedding tensor from safetensors."""
    candidates = ('model.embed_tokens.weight', 'model.model.embed_tokens.weight', 'embed_tokens.weight')
    index = model_dir / 'model.safetensors.index.json'
    found = None
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
        for name in candidates:
            if name in mapping:
                found = (model_dir / mapping[name], name)
                break
    else:
        for shard in sorted(model_dir.glob('*.safetensors')):
            with safe_open(str(shard), framework='pt', device='cpu') as f:
                for name in candidates:
                    if name in f.keys():
                        found = (shard, name)
                        break
            if found:
                break
    if found is None:
        raise RuntimeError('Cannot find Qwen input embedding tensor in safetensors')
    with safe_open(str(found[0]), framework='pt', device='cpu') as f:
        return f.get_tensor(found[1])


def atomic_json(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-config', required=True)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(ROOT / 'configs/default.yaml', resolve(args.model_config), resolve(args.config))
    phys = cfg['physics']
    if phys.get('observable') != 'susceptibility' or phys.get('token_source', 'input_embedding') != 'input_embedding':
        raise ValueError('Expected physics.observable=susceptibility and token_source=input_embedding')
    sizes = sorted(set(map(int, phys['sizes'])))
    if not sizes or sizes[0] <= 0:
        raise ValueError('Invalid physics.sizes')
    device = cfg.get('device', {}).get('device', 'cpu')
    model_dir = resolve(cfg['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    emb = embedding_weights(model_dir).to(device=device, dtype=torch.float32)
    input_path = resolve(cfg['experiment']['input'])
    raw = input_path.read_bytes()
    data = json.loads(raw)
    groups = {}
    for group in data:
        groups.setdefault(float(group['temperature']), []).extend(group['texts'])
    outdir = resolve(cfg['output']['dir'])
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / 'susceptibility_identity.json'
    identity = {'input_sha256': hashlib.sha256(raw).hexdigest(),
                'model_path': str(model_dir), 'sizes': sizes,
                'common_ensemble': bool(phys.get('common_ensemble', True)),
                'method': 'connected_input_embeddings_v1'}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise RuntimeError('Different experiment already owns output directory')
    if not manifest_path.exists():
        atomic_json(manifest_path, identity)
    result_path = outdir / 'susceptibility.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else []
    completed = {(float(r['temperature']), int(r['length'])) for r in results}
    batch_size = int(phys.get('batch_size', 32))
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    for T, texts in sorted(groups.items()):
        ids = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)['input_ids']
        if phys.get('common_ensemble', True):
            ids = [x for x in ids if len(x) >= max(sizes)]
        for L in sizes:
            if (T, L) in completed:
                print(f'[skip] T={T:g} L={L}', flush=True)
                continue
            eligible = [x[:L] for x in ids if len(x) >= L]
            if len(eligible) < 2:
                print(f'[skip] T={T:g} L={L}: insufficient texts', flush=True)
                continue
            # Collect only one D-vector per text; never store [N,L,D].
            sums = []
            for start in range(0, len(eligible), batch_size):
                block = torch.tensor(eligible[start:start + batch_size], device=device)
                with torch.inference_mode():
                    s = torch.nn.functional.embedding(block, emb).float().sum(dim=1)
                sums.append(s.cpu().numpy().astype(np.float64))
            S = np.concatenate(sums)
            center = S.mean(axis=0)
            chi = float(np.square(S - center).sum() / (len(S) * L))
            results.append({'temperature': T, 'length': L, 'n': len(S), 'susceptibility': chi})
            atomic_json(result_path, results)
            print(f'[done] T={T:g} L={L} N={len(S)} chi={chi:.9g}', flush=True)


if __name__ == '__main__':
    main()
