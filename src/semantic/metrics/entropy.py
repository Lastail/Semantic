from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from semantic.data.preprocessing import split_sentences

_CLIENT=None
_MODEL_NAME=None
_TOKENIZER=None
ENTROPY_MAX_WORKERS=50

def configure_entropy(client, model_name, tokenizer, max_workers=50):
    global _CLIENT,_MODEL_NAME,_TOKENIZER,ENTROPY_MAX_WORKERS
    _CLIENT,_MODEL_NAME,_TOKENIZER,ENTROPY_MAX_WORKERS=client,model_name,tokenizer,max_workers

def get_client():
    if _CLIENT is None: raise RuntimeError("Call configure_entropy first")
    return _CLIENT
def get_model_name():
    if _MODEL_NAME is None: raise RuntimeError("Call configure_entropy first")
    return _MODEL_NAME
def get_tokenizer():
    if _TOKENIZER is None: raise RuntimeError("Call configure_entropy first")
    return _TOKENIZER

def tokenize(text, tokenizer):
    """返回不含 BOS 的 token id 列表。"""
    return tokenizer.encode(text, add_special_tokens=False)


def _initial_token_id(tokenizer):
    """选择两端的共同初始条件 token，类比论文 GPT-2 的 <|endoftext|>。

    优先 BOS；若 tokenizer 无 BOS（如 Qwen 系列），退而用 EOS（<|endoftext|>）。
    两端用同一初始 token，边界项（论文 Eq.(15) 首项）可正确纳入并相消。
    """
    for attr in ("bos_token_id", "eos_token_id"):
        tok = getattr(tokenizer, attr, None)
        if tok is not None:
            return tok
    return None


def _prepend_initial(ids, init_tok):
    """在序列最前加初始 token（若存在），保证正/倒序两端用同一初始条件。"""
    return ([init_tok] + list(ids)) if init_tok is not None else list(ids)


def _concat(list_of_lists):
    out = []
    for lst in list_of_lists:
        out.extend(lst)
    return out


def split_sentences(text):
    """按句末标点切分句子（保留标点），丢弃末尾无句末标点的残段（论文的截断策略）。

    注意：对英文缩写/小数点（如 "e.g."、"3.14"）会误切，论文演示用的是普通叙事/陈述句，
    故此处采用简单的逐字符句末标点切分。
    """
    text = (text or "").strip()
    if not text:
        return []
    blocks = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END:
            blocks.append("".join(buf).strip())
            buf = []
    return blocks


# ==================== 核心：NLL 计算 ====================
def _raw_completion_json(client, model_name, ids):
    """用 /v1/completions 一次性取回 prompt 各 token 的条件对数概率的原始 JSON。"""
    resp = client.completions.with_raw_response.create(
        model=model_name,
        prompt=ids,           # token id 列表（vLLM/SGLang 直接按 token id 解析，绕过 chat 模板）
        max_tokens=1,
        echo=True,
        logprobs=1,
        temperature=1.0,      # 打分恒为 temperature=1，取模型原始概率
        extra_body={"prompt_logprobs": 1},
    )
    try:
        return resp.json()
    except AttributeError:
        return json.loads(resp.text)


def _extract_prompt_logprobs(data):
    """从原始 JSON 中取出每个 prompt token 的条件 log p(token_i | token_{<i})。

    返回与 prompt 对齐的 list，首元素为 None（首 token 无条件）。
    """
    choice = data["choices"][0]

    # 首选 vLLM 的 prompt_logprobs：list[ None | list[ {token, token_id, logprob, ...} ] ]
    plp = choice.get("prompt_logprobs")
    if plp is not None:
        out = []
        for entry in plp:
            if not entry:
                out.append(None)
                continue
            if isinstance(entry, dict):
                # vLLM 实际返回 dict[int, Logprob]：键是 token_id，值含 logprob
                top = max(entry.values(),
                      key=lambda x: x["logprob"] if isinstance(x, dict) else getattr(x, "logprob"))
            else:
                # 兼容 list[Logprob] 形式（部分旧版本）
                top = entry[0]
            out.append(top["logprob"] if isinstance(top, dict) else getattr(top, "logprob"))
        return out

    # 回退：echo=True 时 logprobs 覆盖 prompt + 生成 token
    lp = choice.get("logprobs")
    if lp is not None:
        tlp = lp.get("token_logprobs") or []
        # token_logprobs[0] 是首 prompt token（None），末尾一个是生成 token
        return [None] + [v for v in tlp[1:-1]]

    raise RuntimeError(
        "响应中没有 prompt_logprobs / logprobs，无法得到逐 token 对数概率。"
        "请确认服务端（vLLM/SGLang）支持 logprobs，且开启了 /v1/completions 端点。"
    )


def compute_nll(ids, client=None, model_name=None):
    """计算 token id 序列的负对数似然 NLL = -Σ_i log p(ids[i] | ids[:i])。

    返回 (nll, n_used)。ids[0] 视为初始 token，其自身无条件概率不计入求和。
    """
    client = client or get_client()
    model_name = model_name or get_model_name()

    data = _raw_completion_json(client, model_name, ids)
    logps = _extract_prompt_logprobs(data)
    vals = [lp for lp in logps[1:] if lp is not None]
    if not vals:
        raise RuntimeError("未取到任何逐 token 对数概率（序列可能过短）")
    return -sum(vals), len(vals)


# ==================== 熵产生（单条文本） ====================
def entropy_production_single(text, client=None, model_name=None, tokenizer=None, mode="both"):
    """对单条文本计算熵产生，返回 dict。

    返回字段：
      n_tokens              原始文本 token 数（不含 BOS）
      sigma_token           token 级熵产生 σ_token
      sigma_token_per_token σ_token / T
      sigma_block           句子级（block 级）熵产生 σ_block（句子数 <2 时为 None）
      sigma_block_per_token σ_block / T'
      n_blocks              句子数
    """
    tokenizer = tokenizer or get_tokenizer()
    init_tok = _initial_token_id(tokenizer)
    ids = tokenize(text, tokenizer)

    result = {
        "n_tokens": len(ids),
        "sigma_token": None,
        "sigma_token_per_token": None,
        "sigma_block": None,
        "sigma_block_per_token": None,
        "n_blocks": 0,
    }

    # ---- token 级：整条序列完全倒序 ----
    if mode in ("both", "token"):
        fwd_ids = _prepend_initial(ids, init_tok)
        rev_ids = _prepend_initial(list(reversed(ids)), init_tok)
        nll_fwd, n_fwd = compute_nll(fwd_ids, client, model_name)
        nll_rev, _ = compute_nll(rev_ids, client, model_name)
        sigma = nll_rev - nll_fwd
        result["sigma_token"] = float(sigma)
        result["sigma_token_per_token"] = float(sigma / n_fwd) if n_fwd else None

    # ---- 句子级 / block 级：倒句子顺序，截断到最后句末标点 ----
    if mode in ("both", "block"):
        blocks = split_sentences(text)
        result["n_blocks"] = len(blocks)
        if len(blocks) >= 2:
            block_ids = [tokenize(b, tokenizer) for b in blocks]
            fwd_trunc = _prepend_initial(_concat(block_ids), init_tok)
            rev_trunc = _prepend_initial(_concat(list(reversed(block_ids))), init_tok)
            nll_fwd, n_fwd = compute_nll(fwd_trunc, client, model_name)
            nll_rev, _ = compute_nll(rev_trunc, client, model_name)
            sigma = nll_rev - nll_fwd
            result["sigma_block"] = float(sigma)
            result["sigma_block_per_token"] = float(sigma / n_fwd) if n_fwd else None

    return result


# ==================== 批量计算 ====================
def compute_text_entropy(texts, client=None, model_name=None, tokenizer=None, mode="both",
                         max_workers=None, show_progress=True):
    """对一批文本计算熵产生，返回按输入顺序排列的 dict 列表（失败项为 None 并打印原因）。"""
    client = client or get_client()
    model_name = model_name or get_model_name()
    tokenizer = tokenizer or get_tokenizer()
    max_workers = max_workers or ENTROPY_MAX_WORKERS

    if show_progress:
        from tqdm import tqdm
    else:
        def tqdm(x, **kw):  # noqa: ANN001
            return x

    def _work(item):
        idx, text = item
        try:
            return idx, entropy_production_single(text, client, model_name, tokenizer, mode), None
        except Exception as e:  # noqa: BLE001
            return idx, None, str(e)

    results = [None] * len(texts)

    if max_workers <= 1:
        for idx, text in tqdm(list(enumerate(texts)), desc="文本熵产生"):
            _, r, err = _work((idx, text))
            if err:
                print(f"[{idx}] 失败: {err}")
            results[idx] = r
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, (i, t)) for i, t in enumerate(texts)]
            for fut in tqdm(futs, desc="文本熵产生"):
                idx, r, err = fut.result()
                if err:
                    print(f"[{idx}] 失败: {err}")
                results[idx] = r

    return results


# ==================== 按温度扫描 ====================
def load_results(data_path):
    """读取与 units.py 相同的采样 JSON，返回 [{temperature, texts}, ...]。"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("results_prompt", "two_nn_results", "results"):
            if key in data:
                return data[key]
        raise ValueError("JSON dict 中未找到 results_prompt / two_nn_results / results 字段")
    return data


def sweep_temperatures(data_path, out_prefix, mode="both", max_texts=None,
                       max_workers=None, temps=None):
    """按温度计算熵产生，保存聚合 JSON 并绘制 σ/T vs 温度曲线。返回聚合 dict。"""
    entries = load_results(data_path)
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    client = get_client()
    model_name = get_model_name()
    tokenizer = get_tokenizer()

    per_temp = {}
    for entry in entries:
        t = entry["temperature"]
        if temps is not None and not (temps[0] <= t <= temps[1]):
            continue
        texts = entry["texts"]
        per_temp[t] = texts[:max_texts] if max_texts is not None else texts

    temps_sorted = sorted(per_temp.keys())
    summary = []
    for t in temps_sorted:
        print(f"\n=== 温度 {t} ===")
        res = compute_text_entropy(per_temp[t], client, model_name, tokenizer, mode, max_workers)
        summary.append({"temperature": t, "results": res})

    # 聚合：每个温度下的均值/标准差/样本数，以及涨落定理一致性检查
    agg = {}
    for item in summary:
        t = item["temperature"]
        token_vals = [r["sigma_token_per_token"] for r in item["results"]
                      if r and r["sigma_token_per_token"] is not None]
        block_vals = [r["sigma_block_per_token"] for r in item["results"]
                      if r and r["sigma_block_per_token"] is not None]
        token_raw = [r["sigma_token"] for r in item["results"]
                     if r and r["sigma_token"] is not None]

        def _stat(vals):
            if not vals:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

        agg[t] = {
            "token": _stat(token_vals),
            "block": _stat(block_vals),
            # 积分涨落定理一致性检查：E[e^{-σ}] 应 ≈ 1
            "fluctuation_check_token": float(np.mean(np.exp(-np.array(token_raw)))) if token_raw else None,
        }

    out_json = out_prefix + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_json}")

    _plot_entropy_vs_temperature(agg, out_prefix)
    return agg


def _plot_entropy_vs_temperature(agg, out_prefix):
    import matplotlib.pyplot as plt

    temps = sorted(agg.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color in (("token", "token-level σ/T", "tab:purple"),
                              ("block", "block-level σ/T'", "tab:red")):
        xs, ys, es = [], [], []
        for t in temps:
            m = agg[t][key]["mean"]
            if m is None:
                continue
            xs.append(t)
            ys.append(m)
            es.append(agg[t][key]["std"])
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label, color=color)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Entropy production per token")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_pdf = out_prefix + ".pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"曲线已保存: {out_pdf}")


# ==================== 命令行 ====================
