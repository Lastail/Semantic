import httpx
from openai import OpenAI

def make_client(cfg):
    limits = httpx.Limits(max_connections=int(cfg.get("generation", {}).get("max_workers", 50)), max_keepalive_connections=20)
    return OpenAI(base_url=cfg["model"]["endpoint"], api_key=cfg["model"].get("api_key", "EMPTY"), http_client=httpx.Client(limits=limits, timeout=600.0))
