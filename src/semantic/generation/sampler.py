from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from tqdm import tqdm
from transformers import AutoTokenizer
from .client import make_client

def generate_text(client,cfg,prompt,temperature,retries=3):
    last=None
    for attempt in range(retries+1):
        try:
            r=client.chat.completions.create(
                model=cfg["model"]["api_name"],
                messages=[{"role":"user","content":prompt}],
                max_tokens=cfg["generation"]["max_tokens"],
                temperature=temperature,
                extra_body={"chat_template_kwargs":{"enable_thinking":cfg["generation"].get("thinking_enabled",False)}},
            )
            text=r.choices[0].message.content
            return text if isinstance(text,str) else ""
        except Exception as e:
            last=e
            if attempt>=retries: break
            time.sleep(min(2**attempt,8))
    raise RuntimeError(f"Generation failed at T={temperature} after {retries+1} attempts") from last

def _length(tokenizer,text):
    return len(tokenizer(text,add_special_tokens=True,truncation=False)["input_ids"])

def _load_checkpoint(path,tokenizer,min_tokens):
    recovered={}
    if path is None or not path.exists(): return recovered
    rejected={}
    with path.open("r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            try:
                row=json.loads(line)
                T=float(row["temperature"]); text=row["text"]
                if not isinstance(text,str): continue
                if _length(tokenizer,text)>=min_tokens:
                    recovered.setdefault(T,[]).append(text)
                else:
                    rejected[T]=rejected.get(T,0)+1
            except Exception as e:
                print(f"[warning] checkpoint line {line_no}: {e}")
    for T,n in sorted(rejected.items()):
        print(f"[resume] T={T:g}: ignored {n} old samples shorter than {min_tokens}")
    return recovered

def collect_samples(cfg,prompt,temperatures,samples_per_temp=None,checkpoint_path=None,resume=True,retries=3):
    client=make_client(cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg["model"]["path"],trust_remote_code=True)
    g=cfg["generation"]
    target=int(samples_per_temp if samples_per_temp is not None else g["samples_per_temperature"])
    workers=int(g.get("max_workers",50))
    min_tokens=int(g.get("min_tokens",0))
    max_tokens=int(g["max_tokens"])
    checkpoint=Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        if not resume and checkpoint.exists(): checkpoint.unlink()
    recovered=_load_checkpoint(checkpoint,tokenizer,min_tokens) if checkpoint and resume else {}
    lock=Lock(); out=[]
    print(f"[generation] target={target}/T min_tokens={min_tokens} max_tokens={max_tokens} workers={workers}")

    for rawT in temperatures:
        T=float(rawT); texts=list(recovered.get(T,[]))[:target]
        if len(texts)>=target:
            print(f"T={T:g}: recovered {target}/{target}; skip")
            out.append({"temperature":T,"texts":texts}); continue

        short=failed=submitted=0
        print(f"T={T:g}: recovered {len(texts)}/{target}")
        ex=ThreadPoolExecutor(max_workers=workers); futures=set()

        def submit():
            nonlocal submitted
            futures.add(ex.submit(generate_text,client,cfg,prompt,T,retries)); submitted+=1

        for _ in range(min(workers,target-len(texts))): submit()
        bar=tqdm(total=target,initial=len(texts),desc=f"T={T:g} accepted")
        try:
            while len(texts)<target:
                fut=next(as_completed(futures)); futures.remove(fut)
                try: text=fut.result()
                except Exception as e:
                    failed+=1; print(f"\n[warning] T={T:g}: {e}"); text=""

                if text:
                    n_tok=_length(tokenizer,text)
                    if n_tok>=min_tokens:
                        texts.append(text); bar.update(1)
                        if checkpoint:
                            row=json.dumps({"temperature":T,"text":text,"token_length":n_tok},ensure_ascii=False)
                            with lock:
                                with checkpoint.open("a",encoding="utf-8") as f:
                                    f.write(row+"\n"); f.flush()
                    else:
                        short+=1

                if len(texts)<target: submit()
                if submitted%100==0:
                    bar.set_postfix(short=short,failed=failed,requests=submitted)

            for fut in futures: fut.cancel()
        finally:
            bar.close()
            ex.shutdown(wait=False,cancel_futures=True)

        print(f"T={T:g}: accepted={len(texts)}/{target}, short_rejected={short}, failed={failed}, requests={submitted}")
        out.append({"temperature":T,"texts":texts[:target]})
    return out
