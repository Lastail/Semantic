import json, hashlib
from pathlib import Path

def config_hash(cfg): return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
def save_metadata(path,cfg):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"config_hash":config_hash(cfg),"config":cfg},ensure_ascii=False,indent=2),encoding="utf-8")
