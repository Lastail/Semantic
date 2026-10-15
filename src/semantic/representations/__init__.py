from .contextual import ContextualRepresentation
from .positional import PositionalRepresentation
from .hidden_positional import HiddenPositionalRepresentation
from .static_mean import StaticMeanRepresentation
from .contextual_mean import ContextualMeanRepresentation

def build_representation(cfg):
    r=cfg["representation"]; m=cfg["model"]; d=cfg["device"]
    common=dict(model_path=m["path"],device=d["device"],dtype=d.get("dtype","float16"))
    kind=r["type"].lower()
    if kind=="contextual":
        return ContextualRepresentation(**common,layer=r.get("layer",-1),batch_size=r.get("batch_size",4))
    if kind=="positional":
        return PositionalRepresentation(**common,text_block_size=r.get("text_block_size",8),position_block_size=r.get("position_block_size",128))
    if kind == "hidden_positional":
        return HiddenPositionalRepresentation(
            **common,
            layer=r.get("layer", -1),
            batch_size=r.get("batch_size", 4),
            text_block_size=r.get("text_block_size", 8),
            position_block_size=r.get("position_block_size", 128),
            cache_dtype=r.get("cache_dtype", "float32"),
        )
    if kind == "static_mean":
        return StaticMeanRepresentation(
            **common,
            batch_size=r.get("batch_size", 4),
            normalize=r.get("normalize", False),
        )
    if kind == "contextual_mean":
        return ContextualMeanRepresentation(
            **common,
            layer=r.get("layer", -1),
            batch_size=r.get("batch_size", 4),
            normalize=r.get("normalize", False),
        )
    raise ValueError(f"Unknown representation.type={kind!r}; use contextual or positional")
