from .static import StaticEmbeddingExtractor
from .contextual import ContextualExtractor

def build_extractor(cfg):
    r=cfg["representation"]; m=cfg["model"]; d=cfg["device"]
    common=dict(model_path=m["path"],device=d["device"],dtype=d.get("dtype","float16"))
    if r["type"]=="static": return StaticEmbeddingExtractor(**common)
    if r["type"]=="contextual": return ContextualExtractor(**common,layer=r.get("layer",-1),pooling=r.get("pooling","mean"),batch_size=r.get("batch_size",4))
    raise ValueError(r["type"])
