from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .client import make_client

def generate_text(client, cfg, prompt, temperature):
    r = client.chat.completions.create(model=cfg["model"]["api_name"], messages=[{"role":"user","content":prompt}], max_tokens=cfg["generation"]["max_tokens"], temperature=temperature, extra_body={"chat_template_kwargs":{"enable_thinking":cfg["generation"].get("thinking_enabled", False)}})
    return r.choices[0].message.content

def collect_samples(cfg, prompt, temperatures, samples_per_temp=None):
    client = make_client(cfg); n = samples_per_temp or cfg["generation"]["samples_per_temperature"]
    out=[]
    for T in temperatures:
        texts=[]
        with ThreadPoolExecutor(max_workers=cfg["generation"].get("max_workers",50)) as ex:
            fs=[ex.submit(generate_text, client, cfg, prompt, T) for _ in range(n)]
            for f in tqdm(as_completed(fs), total=n, desc=f"T={T:g}"): texts.append(f.result())
        out.append({"temperature":float(T),"texts":texts})
    return out
